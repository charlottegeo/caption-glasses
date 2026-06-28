import argparse
import asyncio

import pygame
import websockets

from config import WEBSOCKET_URI
from client.state import init_state
from client.theme import init_pygame
from client.ui.loop import pygame_loop
from client.ws_client import init_queues, receive_text, send_audio, ws_outbound, ws_sender

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


async def main():
    init_pygame()
    init_state(display_mode=args.mode, caption_telemetry=args.telemetry)
    init_queues()
    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print("Connected to WebSocket.")
            await asyncio.gather(
                send_audio(websocket),
                receive_text(websocket),
                ws_sender(websocket),
                ws_outbound(websocket),
                pygame_loop(websocket),
                return_exceptions=True,
            )
    except Exception as e:
        print(f"Connection Error: {e}")
        pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())
