#!/usr/bin/env python3
"""index.html -> FastAPI -> s1.py (lerobot-rollout / lerobot-replay).

Run this file and open http://127.0.0.1:8000
Install once:  pip install fastapi uvicorn
"""

import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

import s1

# Camera names each policy was trained with (physical cameras are the same).
CAMS_DEFAULT = None                                   # Camara_Alta / camara_pinza
CAMS_OVERVIEW_WRIST = s1.make_cameras("overview", "wrist")

# Endpoint name used by index.html -> (policy folder, camera config)
POLICIES = {
    "moneda": ("PATH_TO_COIN_POLICY", CAMS_DEFAULT),
    "manzana": ("PATH_TO_PICKUP_APPLE_POLICY", CAMS_OVERVIEW_WRIST),
    "pera": ("PATH_TO_PICKUP_PEAR_POLICY", CAMS_DEFAULT),
}

# Dataset folder (with exactly 1 recorded episode) replayed by the dance button.
DANCE_PATH = "PATH_TO_DANCE_DATASET"

REPLAYS = {
    "dance": DANCE_PATH,
}

# Song played by the page during the dance (put the file next to this script).
DANCE_SONG = Path(__file__).parent / "portal.mp3"

app = FastAPI()
robot_lock = threading.Lock()  # the arm does one thing at a time


@app.get("/")
def index():
    html = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/dance-song")
def dance_song():
    if not DANCE_SONG.is_file():
        raise HTTPException(404, f"Song not found: {DANCE_SONG}")
    return FileResponse(DANCE_SONG, media_type="audio/mpeg")


def execute(name, error, run):
    """Run `run()` (blocking) while holding the robot lock."""
    if not robot_lock.acquire(blocking=False):
        raise HTTPException(409, "El robot está ocupado con otro pedido.")
    try:
        if error:
            raise HTTPException(500, error)
        print(f"\n=== {name} ===")
        try:
            return_code, output = run()
        except Exception as exc:
            raise HTTPException(500, f"Fallo al ejecutar: {exc}")
        print(output)
        if return_code != 0:
            raise HTTPException(500, f"El comando terminó con código {return_code}.")
        return {"success": True, "name": name}
    finally:
        robot_lock.release()


# Plain `def`: FastAPI runs these in a worker thread, so the long blocking
# commands don't freeze the server.
@app.post("/api/v1/inferencia/{name}")
def run_inference(name: str):
    if name not in POLICIES:
        raise HTTPException(404, f"Unknown policy '{name}'.")
    policy_path, cameras = POLICIES[name]
    return execute(
        name,
        s1.validate_setup(policy_path),
        lambda: s1.run_rollout_blocking(policy_path, cameras),
    )


@app.post("/api/v1/replay/{name}")
def run_replay(name: str):
    if name not in REPLAYS:
        raise HTTPException(404, f"Unknown replay '{name}'.")
    dataset_path = REPLAYS[name]
    return execute(
        name,
        s1.validate_replay(dataset_path),
        lambda: s1.run_replay_blocking(dataset_path, episode=0),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
