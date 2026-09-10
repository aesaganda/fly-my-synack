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
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from env.loader import list_presets

STATIC = Path(__file__).parent / "static"
JPEG_QUALITY = 70

# Frames are paced on WALL time, not on simulated time.
#
# This used to render every 250 physics steps - 25 ms of simulated time, which
# would be 40 fps if the simulation ran in real time. It does not: it manages
# about 0.16x on this hardware and 0.06x inside a container, so 250 steps took
# most of a second and the browser got 1.5 frames per second. The fly appeared
# to teleport, and no gait survives being sampled like that.
#
# Stepping for a fixed slice of wall time instead gives a smooth stream at
# whatever fraction of real time the machine can manage. The cost is honest
# slow motion, which is reported to the UI rather than hidden.
# Stepping budget per frame. A rendered frame costs 10-12 ms on top, so this is
# also the smoothness/speed dial: at 0.04 the browser gets ~20 fps and the
# renderer takes about a quarter of the wall clock; at 0.02 it gets 35 fps and
# takes half. The simulation itself runs at real time (see README "Real time"),
# so what this trades away is playback speed, and the header states the figure
# it actually achieved rather than claiming one.
FRAME_INTERVAL_S = float(os.environ.get("FLY_FRAME_INTERVAL", "0.05"))
STEP_CHUNK = 10             # steps between clock checks; the check is not free
# Ceiling on playback speed, as a multiple of real time. It exists so a very
# fast machine produces smooth playback rather than half-second jumps between
# frames - but it has to be expressed RELATIVE to real time. A fixed 250 steps
# per frame is 25 ms of simulated time, which at 30 fps caps playback at 0.75x
# however fast the simulation actually runs: once the loop reached real time the
# cap, and not the CPU, was the thing holding it there.
MAX_REALTIME_FACTOR = 1.5

app = FastAPI(title="Connectome fly")

_state: dict = {"session": None, "frame": None, "thread": None, "stop": False,
                "sim_speed": 0.0, "fps": 0.0}
_lock = threading.Lock()


def _camera_res() -> tuple[int, int]:
    """FLY_CAMERA_RES as "HEIGHTxWIDTH", default 360x480.

    Worth turning down in a container. There is no GPU inside one, so MuJoCo
    falls back to software rasterisation and a frame costs 60-100 ms instead of
    ~10 - which comes straight out of the simulation's share of the wall clock
    and shows up as a much larger slow-motion factor. Halving the pixels roughly
    halves the frame cost.
    """
    raw = os.environ.get("FLY_CAMERA_RES", "360x480").lower().replace(",", "x")
    try:
        h, w = (int(v) for v in raw.split("x"))
        return h, w
    except ValueError:
        print(f"warning: FLY_CAMERA_RES={raw!r} is not HxW; using 360x480")
        return 360, 480


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
        start = time.monotonic()
        deadline = start + FRAME_INTERVAL_S
        max_steps = int(MAX_REALTIME_FACTOR * FRAME_INTERVAL_S / sess.body.timestep)
        steps = 0
        with _lock:
            while steps < max_steps and time.monotonic() < deadline:
                for _ in range(STEP_CHUNK):
                    sess.step()
                steps += STEP_CHUNK
            frame = sess.body.render_frame()
        if frame is not None:
            _state["frame"] = _encode_jpeg(frame)
        elapsed = time.monotonic() - start
        if elapsed > 0:
            # Simulated seconds per wall second, and frames per wall second,
            # smoothed - the raw per-frame figure jitters enough to be unreadable.
            a = 0.15
            _state["sim_speed"] += a * (steps * sess.body.timestep / elapsed
                                        - _state["sim_speed"])
            _state["fps"] += a * (1.0 / elapsed - _state["fps"])


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
        # The UI is the demo, so it defaults to the world with something in it.
        # Swapping worlds means recompiling the MuJoCo model and rebuilding the
        # GL context, which cannot be done from the control socket's thread, so
        # this is start-up only - restart with FLY_WORLD=flat for bare ground.
        world=os.environ.get("FLY_WORLD", "room"),
        camera_res=_camera_res(),
        # Real time by default. The controller runs at 333 Hz against the
        # physics at 10 kHz, which is what makes 1x reachable at all - see
        # README "Real time". FLY_CONTROL_EVERY=1 restores the documented model.
        control_every=int(os.environ.get("FLY_CONTROL_EVERY", "30")),
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
            # Poll faster than frames are produced, so the socket adds no
            # latency of its own on top of FRAME_INTERVAL_S.
            await asyncio.sleep(0.015)
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
                                "presets": list_presets(), "brain": sess.neural_frame(),
                                "pace": {"sim_speed": round(_state["sim_speed"], 4),
                                         "fps": round(_state["fps"], 1)}})
            await asyncio.sleep(0.4)

    pusher = asyncio.create_task(push_metrics())
    try:
        while True:
            await _handle(await ws.receive_json(), sess)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        pusher.cancel()
