from __future__ import annotations
import torch
import numpy as np
from numpy import ndarray
from faster_whisper import WhisperModel
from logging import Logger, getLogger

from config import (
    SAMPLE_RATE,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_FINAL_CONDITION_PREVIOUS,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_NOISE_SHIFT,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_PARTIAL_CONDITION_PREVIOUS,
    WHISPER_REPETITION_PENALTY,
    WHISPER_VAD_MIN_SILENCE_MS,
    WHISPER_VAD_SPEECH_PAD_MS,
    WHISPER_VAD_THRESHOLD,
    WHISPER_WITHOUT_TIMESTAMPS,
)
from core.session_settings import SessionSettings

logger: Logger = getLogger(__name__)

device: str = "cuda" if torch.cuda.is_available() else "cpu"
compute_type: str = "float16" if device == "cuda" else "int8"

logger.info("Loading Whisper using %s for transcription.", device.upper())
speech_model: WhisperModel = WhisperModel(
    "Systran/faster-distil-whisper-large-v3",
    device=device,
    compute_type=compute_type,
)

logger.info("Loading VAD via torch hub")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)
vad_model = vad_model.to("cpu")

def _rms(audio: ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def _filter_segments(segments, settings: SessionSettings, is_final: bool) -> str:
    min_lp = (
        settings.segment_min_logprob_final
        if is_final
        else settings.segment_min_logprob_partial
    )
    max_nsp = (
        settings.segment_max_no_speech_final
        if is_final
        else settings.segment_max_no_speech_partial
    )
    kept: list[str] = []
    for seg in segments:
        if seg.avg_logprob < min_lp:
            continue
        if seg.no_speech_prob > max_nsp:
            continue
        t = seg.text.strip()
        if t:
            kept.append(t)
    return " ".join(kept)

def get_speech(
    audio: ndarray,
    is_final: bool = True,
    task: str = "transcribe",
    noise_score: float = 0.0,
    settings: SessionSettings | None = None,
    fresh_context: bool = False,
) -> dict[str, str]:
    """
    Process audio and return transcribed text.
    """
    cfg = settings or SessionSettings()

    if len(audio) < 1600:
        return {"text": ""}

    min_rms = cfg.transcribe_min_rms
    if _rms(audio) < min_rms:
        return {"text": ""}

    if task == "translate":
        beam = 1
        use_previous = False
    elif is_final:
        beam = cfg.final_beam
        use_previous = (
            WHISPER_FINAL_CONDITION_PREVIOUS
            if task == "transcribe" and not fresh_context
            else False
        )
    else:
        beam = cfg.partial_beam
        use_previous = WHISPER_PARTIAL_CONDITION_PREVIOUS if task == "transcribe" else False

    ns = WHISPER_NO_SPEECH_THRESHOLD + WHISPER_NO_SPEECH_NOISE_SHIFT * max(
        0.0, min(1.0, noise_score)
    )
    log_prob = WHISPER_LOG_PROB_THRESHOLD
    compression = WHISPER_COMPRESSION_RATIO_THRESHOLD
    rep_penalty = WHISPER_REPETITION_PENALTY
    if cfg.content_mode == "lyrics":
        ns = float(min(0.92, max(0.55, ns - 0.04)))
        log_prob = -0.85
        compression = 1.35
    ns = float(min(0.99, max(0.5, ns)))

    transcribe_kw: dict = dict(
        beam_size=int(beam),
        language="en" if task == "transcribe" else None,
        task=task,
        condition_on_previous_text=use_previous,
        temperature=0.0,
        vad_filter=cfg.whisper_vad_filter,
        no_speech_threshold=ns,
        log_prob_threshold=log_prob,
        compression_ratio_threshold=compression,
        repetition_penalty=rep_penalty,
    )
    if cfg.whisper_vad_filter:
        transcribe_kw["vad_parameters"] = dict(
            min_silence_duration_ms=WHISPER_VAD_MIN_SILENCE_MS,
            speech_pad_ms=WHISPER_VAD_SPEECH_PAD_MS,
            threshold=WHISPER_VAD_THRESHOLD,
        )
    if WHISPER_WITHOUT_TIMESTAMPS:
        transcribe_kw["without_timestamps"] = True

    try:
        segments, info = speech_model.transcribe(audio, **transcribe_kw)
        text = _filter_segments(list(segments), cfg, is_final)
        lang = ""
        if info is not None and getattr(info, "language", None):
            lang = str(info.language).lower()[:2]
        return {"text": text, "language": lang}
    except (ValueError, RuntimeError):
        return {"text": "", "language": ""}

def check_vad(audio: ndarray) -> float:
    with torch.no_grad():
        sub_chunks = torch.from_numpy(audio).to("cpu").split(512)
        max_prob: float = 0.0
        for sub in sub_chunks:
            if sub.shape[0] < 512:
                sub = torch.nn.functional.pad(sub, (0, 512 - sub.shape[0]))
            prob = vad_model(sub, SAMPLE_RATE).item()
            max_prob = max(max_prob, prob)
        return max_prob