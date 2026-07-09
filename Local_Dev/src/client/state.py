import time
from client.tuning_specs import ALL_SLIDER_SPECS

state: dict = {}
_btn_last_click: dict[str, float] = {}
_meter_cache: dict = {"size": 0, "surf": None}
_last_client_gc = 0.0

_finals_block_cache: list[tuple] = []
_finals_layout_key = None
_partial_cache_key = None
_partial_lines: list[tuple] = []
_partial_height = 0
_caption_lines: list[tuple] = []
_caption_height = 0
_finals_height = 0

def init_state(*, display_mode: str, caption_telemetry: bool) -> None:
    state.clear()
    state.update(
        {
            "display_mode": display_mode,
            "finals": [],
            "partial": {"text": "", "speaker": ""},
            "sound": "",
            "sound_labels": [],
            "sound_timestamp": 0,
            "translate_mode": False,
            "profanity_filter": False,
            "yamnet_profile_index": 0,
            "telemetry": {
                "env": "moderate",
                "noise_score": 0.0,
                "speech_prob": 0.0,
                "input_rms": 0.0,
                "display_noise": 0.0,
                "display_input": 0.0,
                "capturing": False,
                "display_capture": 0.0,
            },
            "caption_telemetry": caption_telemetry,
            "singing_mode": False,
            "show_speakers": True,
            "slider_values": {s["key"]: s["default"] for s in ALL_SLIDER_SPECS},
            "sidebar_scroll": 0,
            "dragging_slider": None,
            "last_caption_mono": 0.0,
            "tuning_tip": "",
        }
    )

def mark_caption_received() -> None:
    state["last_caption_mono"] = time.monotonic()

def invalidate_caption_cache() -> None:
    global _finals_layout_key, _partial_cache_key
    _finals_layout_key = None
    _partial_cache_key = None

def invalidate_partial_cache() -> None:
    global _partial_cache_key
    _partial_cache_key = None

def trim_finals_history() -> None:
    global _finals_block_cache
    from client.tuning_specs import MAX_SENTENCE_HISTORY
    while len(state["finals"]) > MAX_SENTENCE_HISTORY:
        state["finals"].pop(0)
        if _finals_block_cache:
            _finals_block_cache.pop(0)
