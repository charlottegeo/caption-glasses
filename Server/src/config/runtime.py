import os

from config.env import env_bool, env_float, env_int, get_env_variable

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HF_TOKEN: str = get_env_variable("HF_TOKEN", "default")

PHRASE_TIMEOUT: float = env_float("PHRASE_TIMEOUT", "0.75")
MAX_DURATION: float = env_float("MAX_DURATION", "6.0")
VAD_THRESHOLD: float = env_float("VAD_THRESHOLD", "0.4")

SAMPLE_RATE = 16000
CHUNK_SIZE = 2048

PARTIAL_EVERY_N_CHUNKS: int = env_int("PARTIAL_EVERY_N_CHUNKS", "4")
PARTIAL_MIN_INTERVAL_SEC: float = env_float("PARTIAL_MIN_INTERVAL_SEC", "0.2")
LYRICS_PARTIAL_EVERY_N_CHUNKS: int = env_int("LYRICS_PARTIAL_EVERY_N_CHUNKS", "8")
LYRICS_PARTIAL_MIN_INTERVAL_SEC: float = env_float("LYRICS_PARTIAL_MIN_INTERVAL_SEC", "0.45")

SESSION_MAINTENANCE_CHUNKS: int = env_int("SESSION_MAINTENANCE_CHUNKS", "6000")

PROFANITY_FILTER_DEFAULT: bool = env_bool("PROFANITY_FILTER_DEFAULT", "false")
CAPTION_ENV_TELEMETRY: bool = env_bool("CAPTION_ENV_TELEMETRY", "false")