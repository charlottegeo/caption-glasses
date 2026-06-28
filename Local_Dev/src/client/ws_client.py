import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyaudio
import pygame
import websockets
from scipy.signal import resample_poly
from config import DEVICE_CAPTURE_RATE
from client.settings import apply_settings_from_server, update_telemetry
from client.state import invalidate_partial_cache, mark_caption_received, state, trim_finals_history
from client.tuning_specs import WS_AUDIO_QUEUE_MAX, WS_CMD_OUTBOUND_MAX, WS_CMD_QUEUE_MAX
from client.ui.captions import parse_sound_labels

FORMAT = pyaudio.paFloat32
CHANNELS = 1

ws_send_queue: asyncio.Queue | None = None
ws_audio_queue: asyncio.Queue | None = None
ws_cmd_outbound_queue: asyncio.Queue | None = None
_audio_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="caption-audio")

def init_queues() -> None:
    global ws_send_queue, ws_audio_queue, ws_cmd_outbound_queue
    ws_send_queue = asyncio.Queue(maxsize=WS_CMD_QUEUE_MAX)
    ws_audio_queue = asyncio.Queue(maxsize=WS_AUDIO_QUEUE_MAX)
    ws_cmd_outbound_queue = asyncio.Queue(maxsize=WS_CMD_OUTBOUND_MAX)

def ws_send(obj):
    if ws_send_queue is None:
        return
    payload = json.dumps(obj)
    try:
        ws_send_queue.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            ws_send_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            ws_send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


async def ws_outbound(websocket):
    while True:
        payload = None
        try:
            payload = ws_audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        if payload is None:
            audio_task = asyncio.create_task(ws_audio_queue.get())
            cmd_task = asyncio.create_task(ws_cmd_outbound_queue.get())
            done, pending = await asyncio.wait(
                {audio_task, cmd_task}, return_when=asyncio.FIRST_COMPLETED
            )
            payload = next(iter(done)).result()
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await websocket.send(payload)

async def ws_sender(websocket):
    while True:
        msg = await ws_send_queue.get()
        try:
            ws_cmd_outbound_queue.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                ws_cmd_outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                ws_cmd_outbound_queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass

async def send_audio(websocket):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=DEVICE_CAPTURE_RATE,
        input=True,
        frames_per_buffer=4096,
    )
    loop = asyncio.get_running_loop()
    downsample = int(DEVICE_CAPTURE_RATE / 100)

    def read_chunk() -> bytes:
        data = stream.read(4096, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)
        resampled = resample_poly(audio, 160, downsample).astype(np.float32)
        return resampled.tobytes()

    try:
        while True:
            payload = await loop.run_in_executor(_audio_executor, read_chunk)
            await ws_audio_queue.put(payload)
    finally:
        stream.close()
        p.terminate()

async def receive_text(websocket):
    while True:
        try:
            raw = await websocket.recv()
        except websockets.ConnectionClosed:
            break
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        try:
            t = msg.get("type")
            if t == "partial":
                entry = {
                    "text": msg["text"],
                    "speaker": msg.get("speaker", "SPEAKER_00"),
                    "language": msg.get("language"),
                }
                state["partial"] = entry
                mark_caption_received()
                update_telemetry(msg)
            elif t == "final":
                state["finals"].append(
                    {
                        "text": msg["text"],
                        "speaker": msg.get("speaker", "SPEAKER_00"),
                        "language": msg.get("language"),
                    }
                )
                state["partial"] = {"text": "", "speaker": ""}
                invalidate_partial_cache()
                mark_caption_received()
                trim_finals_history()
                update_telemetry(msg)
            elif t == "telemetry":
                update_telemetry(msg)
            elif t == "settings":
                apply_settings_from_server(msg.get("settings", {}))
            elif t == "sound":
                state["sound_labels"] = parse_sound_labels(msg)
                state["sound"] = msg.get("text") or ", ".join(
                    d["label"] for d in state["sound_labels"]
                )
                state["sound_timestamp"] = pygame.time.get_ticks()
        except Exception:
            continue