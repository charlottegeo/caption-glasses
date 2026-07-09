import asyncio
import collections
import functools
import gc
import json
import time
import uuid

import noisereduce as nr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import WebSocket
from logging import Logger, getLogger
from starlette.websockets import WebSocketDisconnect

from config import (
    CAPTION_ENV_TELEMETRY,
    CHUNK_SIZE,
    LYRICS_PARTIAL_EVERY_N_CHUNKS,
    LYRICS_PARTIAL_MIN_INTERVAL_SEC,
    PARTIAL_EVERY_N_CHUNKS,
    PARTIAL_MIN_INTERVAL_SEC,
    PHRASE_TIMEOUT,
    PROFANITY_FILTER_DEFAULT,
    SAMPLE_RATE,
    SESSION_MAINTENANCE_CHUNKS,
)
from core.acoustics import (
    ConnectionAcoustics,
    YamnetAdaptiveState,
    chunk_rms,
    effective_silence_limit,
)
from core.session_settings import MODE_NAMES, SessionSettings
from modules import diarization, profanity_filter, transcription

logger: Logger = getLogger(__name__)
logger.info("Initiating Threads")

CHUNKS_PER_SEC: float = SAMPLE_RATE / CHUNK_SIZE
UTTERANCE_OVERLAP_SEC: float = 0.5
UTTERANCE_OVERLAP_CHUNKS: int = max(1, int(UTTERANCE_OVERLAP_SEC * CHUNKS_PER_SEC))

gpu_lock: asyncio.Lock = asyncio.Lock()
whisper_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
sound_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
diart_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

INT16_FRAME_BYTES: int = CHUNK_SIZE * 2
FLOAT_FRAME_BYTES: int = CHUNK_SIZE * 4

def _normalize_chunk(audio: np.ndarray) -> np.ndarray | None:
    if audio.size == CHUNK_SIZE:
        return audio
    if audio.size > CHUNK_SIZE:
        return audio[:CHUNK_SIZE].copy()
    if audio.size > 0:
        return np.pad(audio.astype(np.float32), (0, CHUNK_SIZE - int(audio.size)))
    return None


def _decode_audio_frame(raw_bytes: bytes) -> np.ndarray | None:
    if not raw_bytes:
        return None
    if len(raw_bytes) == INT16_FRAME_BYTES:
        audio = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
    elif len(raw_bytes) == FLOAT_FRAME_BYTES:
        audio = np.frombuffer(raw_bytes, np.float32).copy()
    elif len(raw_bytes) % 4 == 0 and len(raw_bytes) >= 4:
        audio = np.frombuffer(raw_bytes, np.float32).copy()
        logger.debug("Nonstandard float frame length %s bytes", len(raw_bytes))
    elif len(raw_bytes) % 2 == 0:
        audio = np.frombuffer(raw_bytes, np.int16).astype(np.float32) / 32768.0
        logger.debug("Nonstandard int16 frame length %s bytes", len(raw_bytes))
    else:
        logger.warning("Unrecognized audio frame length %s", len(raw_bytes))
        return None
    return _normalize_chunk(audio)

def _bytes_look_like_json_command(raw: bytes) -> bool:
    s = raw.lstrip()
    return bool(s) and s[:1] == b"{"


def _prepare_audio(
    audio: np.ndarray,
    ac: ConnectionAcoustics,
    settings: SessionSettings,
) -> np.ndarray:
    if settings.skip_denoise:
        gain = ac.post_gain()
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
    reduced = nr.reduce_noise(
        y=audio,
        sr=SAMPLE_RATE,
        prop_decrease=ac.nr_prop_decrease(),
        stationary=ac.nr_stationary(),
    )
    return np.clip(reduced * ac.post_gain(), -1.0, 1.0).astype(np.float32)

def _current_speaker(settings: SessionSettings) -> str:
    return diarization.get_speaker_now(settings.speaker_lookback_sec)


class WebSocketData:
    __slots__ = (
        "connection",
        "uuid",
        "voiced_buffer",
        "is_speaking",
        "is_transcribing",
        "silence_counter",
        "utterance_start_time",
        "pre_roll",
        "yamnet_buffer",
        "chunk_counter",
        "task",
        "profanity_filter_enabled",
        "yamnet_profile",
        "caption_env_telemetry",
        "acoustics",
        "yamnet_adaptive",
        "last_partial_monotonic",
        "settings",
        "last_speech_monotonic",
        "speaker_timeline_reset",
        "lyrics_vad_hangover",
        "partial_transcription_inflight",
        "yamnet_inflight",
        "pending_partial",
        "last_partial_text",
        "last_partial_speaker",
        "last_partial_language",
        "rollover_skip_chunks",
    )

    def __init__(self, websocket: WebSocket, uuid_str: str):
        self.connection: WebSocket = websocket
        self.uuid: str = uuid_str
        self.voiced_buffer: list = []
        self.is_speaking: bool = False
        self.is_transcribing = False
        self.silence_counter: float = 0
        self.utterance_start_time: float = time.monotonic()
        self.pre_roll: collections.deque = collections.deque(maxlen=10)
        self.yamnet_buffer: collections.deque = collections.deque(maxlen=24000)
        self.chunk_counter: int = 0
        self.task: str = "transcribe"
        self.profanity_filter_enabled: bool = PROFANITY_FILTER_DEFAULT
        self.yamnet_profile: str = "default"
        self.caption_env_telemetry: bool = False
        self.acoustics: ConnectionAcoustics = ConnectionAcoustics()
        self.yamnet_adaptive: YamnetAdaptiveState = YamnetAdaptiveState()
        self.last_partial_monotonic: float = 0.0
        self.settings: SessionSettings = SessionSettings(
            phrase_timeout_sec=PHRASE_TIMEOUT
        )
        self.last_speech_monotonic: float = 0.0
        self.speaker_timeline_reset: bool = False
        self.lyrics_vad_hangover: int = 0
        self.partial_transcription_inflight: bool = False
        self.yamnet_inflight: bool = False
        self.pending_partial: tuple[np.ndarray, str] | None = None
        self.last_partial_text: str = ""
        self.last_partial_speaker: str = ""
        self.last_partial_language: str | None = None
        self.rollover_skip_chunks: int = 0


async def handle_client_command(data: dict, client: WebSocketData) -> None:
    if not isinstance(data, dict):
        return
    cmd = data.get("type")
    if cmd == "set_task":
        client.task = str(data.get("value", "transcribe"))
        logger.info("Task updated to: %s", client.task)
    elif cmd == "set_profanity_filter":
        client.profanity_filter_enabled = bool(data.get("value", False))
        logger.info("Profanity filter: %s", client.profanity_filter_enabled)
    elif cmd == "set_yamnet_profile":
        p = data.get("value", "default")
        if isinstance(p, str) and p.strip():
            client.yamnet_profile = p.strip()
        else:
            client.yamnet_profile = "default"
        logger.info("YAMNet profile: %s", client.yamnet_profile)
    elif cmd == "set_caption_telemetry":
        client.caption_env_telemetry = bool(data.get("value", False))
    elif cmd == "set_mode":
        mode = str(data.get("value", "balanced"))
        if mode in MODE_NAMES:
            client.settings.apply_mode(mode)
            logger.info("Session mode: %s", mode)
            await client.connection.send_json(
                {"type": "settings", "mode": mode, "settings": client.settings.as_dict()}
            )
        else:
            logger.warning("Unknown mode %s (valid: %s)", mode, MODE_NAMES)
    elif cmd == "set_settings":
        updates = data.get("value")
        if isinstance(updates, dict):
            applied = client.settings.patch(updates)
            logger.info("Patched settings: %s", applied)
            await client.connection.send_json(
                {"type": "settings", "patched": applied, "settings": client.settings.as_dict()}
            )
    elif cmd == "get_settings":
        await client.connection.send_json(
            {"type": "settings", "settings": client.settings.as_dict()}
        )

def _gpu_cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

async def _session_cleanup(websocket: WebSocketData) -> None:
    logger.info(
        "Session maintenance at chunk %s (client %s)",
        websocket.chunk_counter,
        websocket.uuid,
    )
    diarization.reset_streaming_state()
    websocket.yamnet_adaptive = YamnetAdaptiveState()
    websocket.acoustics.recenter_for_long_session()
    if len(websocket.voiced_buffer) > websocket.rollover_skip_chunks + 4:
        await _flush_utterance(websocket, rolling=False, await_completion=True)
    websocket.voiced_buffer.clear()
    websocket.rollover_skip_chunks = 0
    websocket.is_speaking = False
    websocket.silence_counter = 0
    websocket.lyrics_vad_hangover = 0
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(sound_executor, _gpu_cleanup)

async def _send_telemetry(websocket: WebSocketData, speech_prob: float) -> None:
    if not (CAPTION_ENV_TELEMETRY or websocket.caption_env_telemetry):
        return
    ac = websocket.acoustics
    await websocket.connection.send_json(
        {
            "type": "telemetry",
            "env": ac.env_bucket,
            "noise_score": round(float(ac.noise_score), 3),
            "content_mode": websocket.settings.content_mode,
            "speech_prob": round(float(speech_prob), 3),
            "input_rms": round(float(ac.last_input_rms), 4),
            "capturing": ac.last_gate_open,
            "vad_threshold": round(float(ac.effective_vad_threshold), 3),
        }
    )

def _speech_gate(
    websocket: WebSocketData,
    audio_chunk: np.ndarray,
    speech_prob: float,
) -> bool:
    settings = websocket.settings
    ac = websocket.acoustics
    rms = chunk_rms(audio_chunk)
    is_speech = speech_prob > ac.effective_vad_threshold

    if settings.content_mode == "lyrics":
        energy_floor = max(settings.transcribe_min_rms * 0.85, ac.noise_rms_ema * 0.9)
        has_energy = rms >= energy_floor
        if websocket.is_speaking or websocket.lyrics_vad_hangover > 0:
            if has_energy or speech_prob > 0.08:
                is_speech = True
        elif has_energy and speech_prob > 0.14:
            is_speech = True
        elif speech_prob > ac.effective_vad_threshold:
            is_speech = True
        if is_speech:
            websocket.lyrics_vad_hangover = max(websocket.lyrics_vad_hangover, 10)
        elif websocket.lyrics_vad_hangover > 0:
            is_speech = True
            websocket.lyrics_vad_hangover -= 1
    elif is_speech:
        websocket.lyrics_vad_hangover = 0

    ac.last_gate_open = is_speech
    return is_speech

async def _send_caption_message(
    websocket: WebSocketData,
    msg_type: str,
    text: str,
    speaker: str,
    *,
    language: str | None = None,
) -> None:
    payload: dict = {"type": msg_type, "text": text, "speaker": speaker}
    if language and websocket.task == "translate" and language != "en":
        payload["language"] = language
    if CAPTION_ENV_TELEMETRY or websocket.caption_env_telemetry:
        payload["env"] = websocket.acoustics.env_bucket
        payload["noise_score"] = round(float(websocket.acoustics.noise_score), 3)
        payload["content_mode"] = websocket.settings.content_mode
    await websocket.connection.send_json(payload)

async def create_connection(websocket: WebSocket) -> None:
    generated_uuid: str = str(uuid.uuid4())
    logger.info("Established connection with Client %s", generated_uuid)

    client_connection = WebSocketData(websocket, generated_uuid)
    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                raw = message["bytes"]
                if _bytes_look_like_json_command(raw):
                    try:
                        data = json.loads(raw.decode("utf-8"))
                        if isinstance(data, dict) and data.get("type") is not None:
                            await handle_client_command(data, client_connection)
                            continue
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                await process_websocket_bytes(raw, client_connection)

            elif message.get("text") is not None:
                try:
                    data = json.loads(message["text"])
                    await handle_client_command(data, client_connection)
                except json.JSONDecodeError as e:
                    logger.error("Error parsing command: %s", e)
    except WebSocketDisconnect:
        logger.info("Disconnected with Client %s", generated_uuid)
    except Exception as e:
        logger.error("Error: %s: %s", type(e).__name__, e)

async def process_audio_task(
    websocket: WebSocketData,
    audio_data: np.ndarray,
    speaker_at_capture: str,
    is_final: bool = True,
    *,
    fresh_context: bool = False,
) -> None:
    if audio_data is None or len(audio_data) == 0:
        return

    loop = asyncio.get_running_loop()
    ac = websocket.acoustics
    settings = websocket.settings
    denoise_fn = functools.partial(_prepare_audio, audio_data, ac, settings)
    clean_audio = await loop.run_in_executor(sound_executor, denoise_fn)

    async with gpu_lock:
        websocket.is_transcribing = True
        try:
            result = await loop.run_in_executor(
                whisper_executor,
                transcription.get_speech,
                clean_audio,
                is_final,
                websocket.task,
                ac.noise_score,
                settings,
                fresh_context,
            )
            text = result["text"].strip()
            lang = result.get("language") or None
            if is_final and not text and websocket.last_partial_text.strip():
                text = websocket.last_partial_text.strip()
                speaker_at_capture = websocket.last_partial_speaker or speaker_at_capture
                lang = websocket.last_partial_language or lang
                logger.debug("Sending last partial as final (%d chars)", len(text))
            if text:
                text = profanity_filter.mask_caption_text(
                    text, websocket.profanity_filter_enabled
                )
                if not text:
                    return
                msg_type = "final" if is_final else "partial"
                if not is_final:
                    websocket.last_partial_text = text
                    websocket.last_partial_speaker = speaker_at_capture
                    websocket.last_partial_language = lang
                else:
                    websocket.last_partial_text = ""
                    websocket.last_partial_speaker = ""
                    websocket.last_partial_language = None
                await _send_caption_message(
                    websocket, msg_type, text, speaker_at_capture, language=lang
                )
        except Exception as e:
            logger.exception("Transcription Error: %s", e)
        finally:
            websocket.is_transcribing = False


def _max_utterance_chunks(settings: SessionSettings) -> int:
    return max(2, int(settings.max_utterance_sec * CHUNKS_PER_SEC))


def _slice_voiced_audio(
    websocket: WebSocketData,
    *,
    end: int | None = None,
) -> np.ndarray | None:
    buf = websocket.voiced_buffer
    start = websocket.rollover_skip_chunks
    stop = len(buf) if end is None else end
    if stop <= start + 4:
        return None
    return np.concatenate(buf[start:stop])


def _partial_cadence(settings: SessionSettings) -> tuple[int, float]:
    if settings.content_mode == "lyrics":
        return max(4, LYRICS_PARTIAL_EVERY_N_CHUNKS), LYRICS_PARTIAL_MIN_INTERVAL_SEC
    return max(1, PARTIAL_EVERY_N_CHUNKS), PARTIAL_MIN_INTERVAL_SEC


async def _flush_pending_partial(websocket: WebSocketData) -> None:
    try:
        while websocket.pending_partial is not None:
            audio_data, speaker = websocket.pending_partial
            websocket.pending_partial = None
            await process_audio_task(
                websocket, audio_data, speaker, is_final=False
            )
    finally:
        websocket.partial_transcription_inflight = False
        if websocket.pending_partial is not None:
            websocket.partial_transcription_inflight = True
            asyncio.create_task(_flush_pending_partial(websocket))

async def _schedule_partial_transcription(
    websocket: WebSocketData,
    audio_data: np.ndarray,
    speaker: str,
    *,
    is_final: bool,
    fresh_context: bool = False,
) -> None:
    if not is_final:
        websocket.pending_partial = (audio_data.copy(), speaker)
        if websocket.partial_transcription_inflight:
            return
        websocket.partial_transcription_inflight = True
        asyncio.create_task(_flush_pending_partial(websocket))
        return

    websocket.pending_partial = None
    asyncio.create_task(
        process_audio_task(
            websocket,
            audio_data,
            speaker,
            is_final=True,
            fresh_context=fresh_context,
        )
    )

async def _flush_utterance(
    websocket: WebSocketData,
    *,
    rolling: bool = False,
    overlap_chunks: int | None = None,
    await_completion: bool = False,
) -> None:
    overlap = overlap_chunks if overlap_chunks is not None else UTTERANCE_OVERLAP_CHUNKS
    if rolling:
        commit_end = len(websocket.voiced_buffer) - overlap
        audio = _slice_voiced_audio(websocket, end=commit_end)
    else:
        audio = _slice_voiced_audio(websocket)
    if audio is None:
        return
    speaker = _current_speaker(websocket.settings)
    task = process_audio_task(
        websocket,
        audio,
        speaker,
        is_final=True,
        fresh_context=rolling,
    )
    if await_completion:
        await task
    else:
        asyncio.create_task(task)
    if rolling:
        keep = max(1, overlap)
        websocket.voiced_buffer = list(websocket.voiced_buffer[-keep:])
        websocket.rollover_skip_chunks = len(websocket.voiced_buffer)
    else:
        websocket.voiced_buffer = []
        websocket.rollover_skip_chunks = 0
        websocket.is_speaking = False
        websocket.pre_roll.clear()

async def process_speaking(websocket: WebSocketData, audio_chunk: np.ndarray) -> None:
    now = time.monotonic()
    websocket.last_speech_monotonic = now
    websocket.speaker_timeline_reset = False

    if not websocket.is_speaking:
        websocket.is_speaking = True
        websocket.utterance_start_time = now
        websocket.rollover_skip_chunks = 0
        websocket.last_partial_text = ""
        websocket.last_partial_speaker = ""
        websocket.last_partial_language = None
        websocket.voiced_buffer.extend(list(websocket.pre_roll))
    websocket.voiced_buffer.append(audio_chunk)
    websocket.silence_counter = 0

    n, partial_interval = _partial_cadence(websocket.settings)
    new_audio_chunks = len(websocket.voiced_buffer) - websocket.rollover_skip_chunks
    if (
        new_audio_chunks >= n
        and new_audio_chunks % n == 0
    ):
        if (
            partial_interval <= 0.0
            or (now - websocket.last_partial_monotonic) >= partial_interval
        ):
            audio = _slice_voiced_audio(websocket)
            if audio is not None:
                websocket.last_partial_monotonic = now
                await _schedule_partial_transcription(
                    websocket,
                    audio,
                    _current_speaker(websocket.settings),
                    is_final=False,
                )

    if len(websocket.voiced_buffer) >= _max_utterance_chunks(websocket.settings):
        overlap = UTTERANCE_OVERLAP_CHUNKS
        if websocket.settings.content_mode == "lyrics":
            overlap = max(overlap, int(1.2 * CHUNKS_PER_SEC))
        await _flush_utterance(websocket, rolling=True, overlap_chunks=overlap)

async def process_silence(websocket: WebSocketData, audio_chunk: np.ndarray) -> None:
    now = time.monotonic()

    if websocket.is_speaking:
        websocket.voiced_buffer.append(audio_chunk)
        websocket.silence_counter += 1
        limit = effective_silence_limit(
            websocket.acoustics,
            websocket.settings.phrase_timeout_sec,
            CHUNKS_PER_SEC,
            content_mode=websocket.settings.content_mode,
        )
        if websocket.silence_counter > limit:
            websocket.is_speaking = False
            await _flush_utterance(websocket, rolling=False)
    else:
        websocket.pre_roll.append(audio_chunk)
        if websocket.last_speech_monotonic > 0 and not websocket.speaker_timeline_reset:
            silent_for = now - websocket.last_speech_monotonic
            if silent_for >= websocket.settings.speaker_reset_sec:
                diarization.reset_speaker_timeline()
                websocket.speaker_timeline_reset = True
                logger.debug("Speaker timeline reset after %.1fs silence", silent_for)

async def process_websocket_bytes(raw_bytes: bytes, websocket: WebSocketData) -> None:
    loop = asyncio.get_running_loop()

    audio_chunk = _decode_audio_frame(raw_bytes)
    if audio_chunk is None:
        return

    loop.run_in_executor(
        diart_executor, diarization.audio_source.push_audio, audio_chunk.copy()
    )

    websocket.yamnet_buffer.extend(audio_chunk)
    websocket.chunk_counter += 1

    if websocket.chunk_counter % 8 == 0 and len(websocket.yamnet_buffer) == 24000:
        if not websocket.yamnet_inflight:

            async def ps(buf: np.ndarray, ws: WebSocketData) -> None:
                ws.yamnet_inflight = True
                try:
                    exclude = {"music"} if ws.settings.content_mode == "lyrics" else None
                    fn = functools.partial(
                        diarization.get_sounds,
                        buf,
                        profile=ws.yamnet_profile,
                        adaptive=ws.yamnet_adaptive,
                        exclude_categories=exclude,
                        settings=ws.settings,
                    )
                    detected = await loop.run_in_executor(sound_executor, fn)
                    if detected:
                        combined_text = ", ".join(d["label"] for d in detected)
                        await ws.connection.send_json(
                            {
                                "type": "sound",
                                "text": combined_text,
                                "labels": detected,
                                "speaker": "",
                            }
                        )
                except Exception as e:
                    logger.error("YAMNet Error: %s", e)
                finally:
                    ws.yamnet_inflight = False

            asyncio.create_task(ps(np.array(websocket.yamnet_buffer, copy=True), websocket))

    speech_prob: float = await loop.run_in_executor(
        sound_executor, transcription.check_vad, audio_chunk
    )
    websocket.acoustics.update(audio_chunk, speech_prob, websocket.settings)

    is_speech = _speech_gate(websocket, audio_chunk, speech_prob)

    if (
        SESSION_MAINTENANCE_CHUNKS > 0
        and websocket.chunk_counter % SESSION_MAINTENANCE_CHUNKS == 0
    ):
        asyncio.create_task(_session_maintenance(websocket))

    if websocket.chunk_counter % 4 == 0 and (
        CAPTION_ENV_TELEMETRY or websocket.caption_env_telemetry
    ):
        await _send_telemetry(websocket, speech_prob)

    if is_speech:
        await process_speaking(websocket, audio_chunk)
    else:
        await process_silence(websocket, audio_chunk)