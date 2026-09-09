"""FastAPI wrapper around a running simulation.

Read-only sim control. This service is never given NEUPRINT_TOKEN and never
writes to the connectome volume - see docker-compose.yml, where /data/connectome
is mounted read-only here.

Two sockets rather than one: a slow control message must not stall the video,
and a dropped frame must not lose a command.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from env.loader import list_presets

STATIC = Path(__file__).parent / "static"
RENDER_EVERY = 250          # physics steps between rendered frames (~40 fps of sim time)
JPEG_QUALITY = 70

app = FastAPI(title="Connectome fly")

_state: dict = {"session": None, "frame": None, "thread": None, "stop": False}
_lock = threading.Lock()


def _encode_jpeg(rgb) -> bytes:
    from PIL import Image  # pillow arrives with flygym; no extra dependency

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _sim_loop() -> None:
    sess = _state["session"]
    while not _state["stop"]:
        if sess.paused:
            threading.Event().wait(0.05)
            continue
        with _lock:
            for _ in range(RENDER_EVERY):
                sess.step()
            frame = sess.body.render_frame()
        if frame is not None:
            _state["frame"] = _encode_jpeg(frame)


@app.on_event("startup")
def _startup() -> None:
    from session import Session

    sess = Session(
        preset=os.environ.get("FLY_ENV", "dry_land"),
        connectome=os.environ.get("FLY_CONNECTOME", "synthetic"),
        subset=os.environ.get("FLY_SUBSET", "motor"),
        device=os.environ.get("FLY_DEVICE", "cpu"),
        render=True,
        connectome_dir=os.environ.get("FLY_CONNECTOME_DIR", "/data/connectome"),
        synthetic_size=int(os.environ.get("FLY_SYNTHETIC_SIZE", "20000")),
    )
    _state["session"] = sess
    t = threading.Thread(target=_sim_loop, daemon=True)
    t.start()
    _state["thread"] = t


@app.on_event("shutdown")
def _shutdown() -> None:
    _state["stop"] = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz() -> dict:
    sess = _state["session"]
    return {"ok": sess is not None, "step": sess.step_count if sess else 0}


@app.get("/metrics")
def metrics() -> dict:
    sess = _state["session"]
    return sess.live_metrics() if sess else {}


@app.get("/stream.mjpg")
def mjpeg() -> StreamingResponse:
    """No-JS fallback for the WebSocket frame feed."""

    def gen():
        import time

        while True:
            frame = _state["frame"]
            if frame:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.websocket("/ws/frames")
async def ws_frames(ws: WebSocket) -> None:
    await ws.accept()
    last = None
    try:
        while True:
            frame = _state["frame"]
            if frame is not None and frame is not last:
                await ws.send_bytes(frame)
                last = frame
            await asyncio.sleep(0.04)
    except (WebSocketDisconnect, RuntimeError):
        return


# Commands accepted from the browser. Deliberately a closed set - anything not
# listed here is ignored rather than dispatched dynamically.
async def _handle(cmd: dict, sess) -> None:
    name = cmd.get("cmd")
    if name == "preset" and cmd.get("name") in list_presets():
        with _lock:
            sess.switch_preset(cmd["name"])
    elif name == "pause":
        sess.paused = True
    elif name == "resume":
        sess.paused = False
    elif name == "reset":
        with _lock:
            sess.reset()
    elif name == "override":
        sess.decoder.set_override(
            **{k: v for k, v in cmd.items() if k in ("forward", "turn", "stop")}
        )
    elif name == "clear_override":
        sess.decoder.clear_override()


@app.websocket("/ws/control")
async def ws_control(ws: WebSocket) -> None:
    await ws.accept()
    sess = _state["session"]
    await ws.send_json({"presets": list_presets(), "preset": sess.preset.name})

    async def push_metrics() -> None:
        while True:
            await ws.send_json({"metrics": sess.live_metrics(), "preset": sess.preset.name,
                                "presets": list_presets()})
            await asyncio.sleep(0.4)

    pusher = asyncio.create_task(push_metrics())
    try:
        while True:
            await _handle(await ws.receive_json(), sess)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        pusher.cancel()
