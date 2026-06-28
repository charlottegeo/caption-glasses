import time
import warnings
import numpy as np
import collections
from logging import Logger, getLogger
from pyannote.core import Annotation, Segment
import noisereduce as nr

import tensorflow as tf
#this makes tensorflow use CPU only
tf.config.set_visible_devices([], 'GPU')
tf.config.optimizer.set_jit(False)

import tensorflow_hub as hub

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import AudioSource
from diart.inference import StreamingInference

from config import (
    HF_TOKEN,
    SAMPLE_RATE,
    YAMNET_BASE_THRESHOLD,
    YAMNET_MAX_LABELS,
    YAMNET_MUSIC_THRESHOLD,
    YAMNET_NATURAL_THRESHOLD,
    YAMNET_PRESETS_MERGED,
    YAMNET_SECONDARY_MIN_RATIO,
    YAMNET_TOP_K,
    YAMNET_VEHICLE_THRESHOLD,
)
from core.acoustics import YamnetAdaptiveState
from core.session_settings import SessionSettings

yamnet_model: hub.KerasLayer = hub.load("https://tfhub.dev/google/yamnet/1")
YAMNET_CATEGORY_RANGES: dict[str, range] = {
    "human": range(0, 67),
    "animal": range(67, 132),
    "music": range(132, 277),
    "natural": range(277, 294),
    "vehicle": range(294, 348),
    "domestic": range(348, 412),
    "tools": range(412, 420),
    "explosive": range(420, 456),
    "misc": range(456, 521),
}

def _category_for_index(idx: int) -> str:
    for cat_name, cat_range in YAMNET_CATEGORY_RANGES.items():
        if idx in cat_range:
            return cat_name
    return "misc"

class_map_path: bytes = yamnet_model.class_map_path().numpy()
class_names: list[str] = []

with tf.io.gfile.GFile(class_map_path) as f:
    class_names = [
        line.split(",")[2].strip().strip('"') for line in f.read().splitlines()[1:]
    ]

class WebSocketAudioSource(AudioSource):
    def __init__(self, sample_rate):
        super().__init__(uri="websocket_stream", sample_rate=sample_rate)

    def read(self):
        pass

    def close(self):
        self.stream.on_completed()

    def push_audio(self, chunk: np.ndarray):
        self.stream.on_next(chunk.reshape(1, -1))

logger: Logger = getLogger(__name__)
logger.info("Loading Diart (Pyannote)...")
warnings.filterwarnings(
    "ignore",
    message="Mismatch between frames .* and weights .* numbers.",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)

diart_config: SpeakerDiarizationConfig = SpeakerDiarizationConfig(
    duration=5.0,
    step=0.5,
    latency="min",
    sample_rate=SAMPLE_RATE,
    hf_token=HF_TOKEN,
    tau_active=0.5,
)
diarization: SpeakerDiarization = SpeakerDiarization(diart_config)

audio_source: WebSocketAudioSource = WebSocketAudioSource(SAMPLE_RATE)
pipeline: StreamingInference = StreamingInference(diarization, audio_source)


speaker_timeline: collections.deque[tuple[float, str]] = collections.deque(
    maxlen=200
)

def on_diarization_update(result: tuple[Annotation] | Annotation) -> None:
    """
    Adds result processed from the Audio Pipeline to be added to the speaker queue

    Arguments:
        result (tuple[Annotation] | Annotation): Either a tuple or just an annotation, depending on current buffer
    """

    annotation: Annotation = result[0] if isinstance(result, tuple) else result

    if not hasattr(annotation, "labels") or not annotation.labels():
        return

    try:
        tracks: list[tuple[Segment, str, str]] = list(
            annotation.itertracks(yield_label=True)
        )
        if not tracks:
            return
        latest_track: tuple[Segment, str, str] = max(tracks, key=lambda x: x[0].end)
        speaker_timeline.append((time.monotonic(), latest_track[2]))
    except Exception:
        return

def _yamnet_thresholds_for_profile(profile: str) -> dict[str, float]:
    base = {
        "base_threshold": YAMNET_BASE_THRESHOLD,
        "vehicle_threshold": YAMNET_VEHICLE_THRESHOLD,
        "natural_threshold": YAMNET_NATURAL_THRESHOLD,
        "music_threshold": YAMNET_MUSIC_THRESHOLD,
        "secondary_min_ratio": YAMNET_SECONDARY_MIN_RATIO,
        "top_k": float(YAMNET_TOP_K),
        "max_labels": float(YAMNET_MAX_LABELS),
    }
    preset = YAMNET_PRESETS_MERGED.get(profile) or YAMNET_PRESETS_MERGED.get("default") or {}
    for key, delta in preset.items():
        if key in base:
            base[key] = float(base[key]) + float(delta)
        else:
            base[key] = float(delta)
    base["secondary_min_ratio"] = float(np.clip(base["secondary_min_ratio"], 0.5, 0.99))
    base["base_threshold"] = float(np.clip(base["base_threshold"], 0.2, 0.95))
    base["vehicle_threshold"] = float(np.clip(base["vehicle_threshold"], 0.2, 0.95))
    base["natural_threshold"] = float(np.clip(base["natural_threshold"], 0.2, 0.95))
    base["music_threshold"] = float(np.clip(base["music_threshold"], 0.2, 0.95))
    return base

def _resolve_yamnet_thresholds(
    profile: str,
    settings: SessionSettings | None,
) -> dict[str, float]:
    thr = _yamnet_thresholds_for_profile(profile)
    if settings is None:
        return thr
    thr["base_threshold"] = float(settings.yamnet_base_threshold)
    thr["music_threshold"] = float(settings.yamnet_music_threshold)
    thr["secondary_min_ratio"] = float(settings.yamnet_secondary_min_ratio)
    thr["max_labels"] = float(settings.yamnet_max_labels)
    thr["secondary_min_ratio"] = float(np.clip(thr["secondary_min_ratio"], 0.5, 0.99))
    thr["base_threshold"] = float(np.clip(thr["base_threshold"], 0.2, 0.95))
    thr["music_threshold"] = float(np.clip(thr["music_threshold"], 0.2, 0.95))
    return thr

def get_sounds(
    audio: np.ndarray,
    profile: str = "default",
    adaptive: YamnetAdaptiveState | None = None,
    exclude_categories: set[str] | None = None,
    settings: SessionSettings | None = None,
) -> list[dict[str, str | float]]:
    """
    Processes audio for sounds using category-based deduplication.
    Returns dicts: label, category, score.
    """
    thr = _resolve_yamnet_thresholds(profile, settings)
    base_threshold = thr["base_threshold"]
    vehicle_threshold = thr["vehicle_threshold"]
    natural_threshold = thr["natural_threshold"]
    music_threshold = thr["music_threshold"]
    secondary_min_ratio = thr["secondary_min_ratio"]
    top_k = int(thr.get("top_k", YAMNET_TOP_K))
    max_labels = int(thr.get("max_labels", YAMNET_MAX_LABELS))
    skip_cats = exclude_categories or set()
    denoise_strength = 0.45
    if settings is not None:
        denoise_strength = float(np.clip(settings.yamnet_denoise_strength, 0.0, 0.95))

    floor_delta = 0.0
    if adaptive is not None:
        floor_delta = adaptive.take_floor_delta()

    base_threshold += floor_delta
    vehicle_threshold += floor_delta
    natural_threshold += floor_delta

    if denoise_strength > 0.02:
        clean_audio = nr.reduce_noise(
            y=audio,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=denoise_strength,
        )
    else:
        clean_audio = audio
    scores, _, _ = yamnet_model(clean_audio)
    class_scores: tf.Tensor = tf.reduce_max(scores, axis=0)

    top_indices = tf.argsort(class_scores, direction="DESCENDING")[:top_k].numpy()

    candidates = []
    for idx in top_indices:
        label = class_names[idx]
        score = float(class_scores[idx].numpy())
        assigned_cat = _category_for_index(int(idx))
        if assigned_cat in skip_cats:
            continue

        th = base_threshold
        if assigned_cat == "vehicle":
            th = vehicle_threshold
        elif assigned_cat == "natural":
            th = natural_threshold
        elif assigned_cat == "music":
            th = music_threshold

        if label in ["Silence", "Speech"] or score < th:
            continue

        candidates.append({"label": label, "score": score, "category": assigned_cat})

    best_per_category: dict[str, dict] = {}
    for cand in candidates:
        cat = cand["category"]
        if cat not in best_per_category or cand["score"] > best_per_category[cat]["score"]:
            best_per_category[cat] = cand

    if not best_per_category:
        return []

    global_top = max(c["score"] for c in best_per_category.values())
    floor = global_top * secondary_min_ratio

    kept = [c for c in best_per_category.values() if c["score"] >= floor]
    kept.sort(key=lambda x: x["score"], reverse=True)
    out = [
        {
            "label": c["label"],
            "category": c["category"],
            "score": round(float(c["score"]), 3),
        }
        for c in kept[:max_labels]
    ]
    if adaptive is not None and best_per_category:
        adaptive.after_window(global_top)
    return out

def reset_speaker_timeline() -> None:
    speaker_timeline.clear()

def reset_streaming_state() -> None:
    speaker_timeline.clear()
    for target in (pipeline, diarization):
        reset = getattr(target, "reset", None)
        if callable(reset):
            try:
                reset()
                return
            except Exception:
                pass

def get_speaker_now(lookback_sec: float = 0.35) -> str:
    now = time.monotonic()
    for ts, spk in reversed(speaker_timeline):
        if now - ts <= lookback_sec:
            return spk
    return "SPEAKER_00"

def get_speaker_at(timestamp: float, max_age: float = 1.0) -> str:
    """
    Finds the most recent speaker at or before the given timestamp, or 00 if none is found

    Arguments:
        timestamp (float): The timestamp of the speaker to look for
        max_age (float): The max age to look back, defaults to 1.5

    Returns:
        str: The label of the speaker, or SPEAKER_00 if none is found
    """

    best: str | None = None

    for ts, spk in reversed(speaker_timeline):
        if timestamp - 1.0 <= ts <= timestamp + max_age:
            return spk
        if ts <= timestamp + max_age:
            best = spk
            break
    return best or "SPEAKER_00"

pipeline.stream.subscribe(on_diarization_update)