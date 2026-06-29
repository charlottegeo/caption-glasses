import argparse
import asyncio
import sys

import pygame
import websockets

from config import WEBSOCKET_URI
from client.state import init_state
from client.theme import init_pygame
from client.ui.loop import paint_startup_frame, run_pygame_loop
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


async def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                print(f"Background task {i} failed: {type(result).__name__}: {result}")


async def main():
    init_pygame()
    init_state(display_mode=args.mode, caption_telemetry=args.telemetry)
    init_queues()
    paint_startup_frame()

    bg_tasks: list[asyncio.Task] = []
    try:
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            print("Connected to WebSocket.")
            bg_tasks = [
                asyncio.create_task(send_audio(websocket)),
                asyncio.create_task(receive_text(websocket)),
                asyncio.create_task(ws_sender(websocket)),
                asyncio.create_task(ws_outbound(websocket)),
            ]
            await run_pygame_loop()
    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        await _cancel_tasks(bg_tasks)
        pygame.quit()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
