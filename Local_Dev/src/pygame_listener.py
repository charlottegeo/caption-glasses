import argparse
import time

import pygame

from config import WEBSOCKET_URI
from client.state import init_state
from client.theme import init_pygame
from client.ui.loop import paint_startup_frame, run_pygame_loop
from client.ws_client import (
    init_queues,
    network_connected,
    network_error,
    start_network,
    stop_network,
)

parser = argparse.ArgumentParser(description="Transcription Display")
parser.add_argument(
    "--mode",
    choices=["label", "color"],
    default="label",
    help="Display mode: 'label' shows [Speaker], 'color' relies on text color.",
)
parser.add_argument(
    "--telemetry",
    action="store_true",
    help="Send set_caption_telemetry so server includes env/noise_score on captions.",
)
args = parser.parse_args()

CONNECT_TIMEOUT_SEC = 30.0


def _wait_for_connection() -> bool:
    deadline = time.monotonic() + CONNECT_TIMEOUT_SEC
    while time.monotonic() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        err = network_error()
        if err is not None:
            paint_startup_frame(f"Connection failed: {err}")
            return False
        if network_connected():
            return True
        paint_startup_frame()
        time.sleep(0.05)
    paint_startup_frame("Connection timed out")
    return False


def main() -> None:
    init_pygame()
    init_state(display_mode=args.mode, caption_telemetry=args.telemetry)
    init_queues()
    paint_startup_frame()

    start_network(WEBSOCKET_URI)
    try:
        if not _wait_for_connection():
            time.sleep(1.5)
            return
        run_pygame_loop()
    finally:
        stop_network()
        pygame.quit()


if __name__ == "__main__":
    main()
