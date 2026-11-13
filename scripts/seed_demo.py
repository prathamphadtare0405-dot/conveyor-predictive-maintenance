"""Replay a simulated belt loop through the vision API, to exercise the damage registry.

    python scripts/seed_demo.py            # backend must be running, vision model trained

READ THIS FIRST. The dataset is unrelated still photographs of cracks in roads and walls.
It is not footage of one belt, and no public multi-lap belt footage was available. So the
loop is simulated:

* A handful of dataset images (from the test split, which the model never saw in
  training or validation) are each given a fixed position on a 10 km belt loop. Each one
  stands for "one damage on the belt".
* The belt is then "run" for several laps. Every lap the camera passes each position and
  the image placed there is sent to the backend again, changed a little each time
  (brightness, a small shift, sometimes blur), the way the same damage would look
  different a lap later. Clean frames (plain dark texture) are sent in between.
* The camera's position estimate comes from a tracker with a speed sensor that reads
  slightly wrong, corrected once per lap by a reference marker. The marker event is
  given to the tracker; it is not detected from an image.

Scripted events, so the registry has something to show:

* one damage grows (its image is zoomed in a little more each lap),
* one damage is "repaired" (stops being shown after lap 2),
* one damage appears late (first shown on lap 3),
* two damages sit 3 m apart, inside the matching tolerance.

What this proves: the registry logic (one damage counted once across laps, neighbours
kept apart, growth tracked, missing damage marked). What it does not prove: that
appearance matching survives real dust, lighting and wear, or that position tracking
works with a real encoder. See the README.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from vision.belt_registry import BeltTracker  # noqa: E402

TEST_IMAGES = settings.data_dir / "datasets" / "crack-seg" / "images" / "test"
SPEED_BIAS = 0.004  # the speed sensor reads 0.4 % high all the time...
SPEED_JITTER = 0.0002  # ...and wobbles by another 0.02 % from lap to lap


@dataclass
class PlacedDamage:
    name: str
    image: np.ndarray
    position_m: float
    first_lap: int = 1
    last_lap: int = 10**9
    grows: bool = False


def vary(image: np.ndarray, rng: np.random.Generator, zoom: float = 1.0) -> np.ndarray:
    """The same damage a lap later: different light, slightly shifted, sometimes less sharp."""
    h, w = image.shape[:2]
    out = image
    if zoom > 1.0:  # growth: the damage fills more of the frame
        ch, cw = int(h / zoom), int(w / zoom)
        top, left = (h - ch) // 2, (w - cw) // 2
        out = cv2.resize(out[top : top + ch, left : left + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    shift = np.float32([[1, 0, rng.integers(-6, 7)], [0, 1, rng.integers(-6, 7)]])
    out = cv2.warpAffine(out, shift, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = cv2.convertScaleAbs(out, alpha=float(rng.uniform(0.88, 1.12)), beta=float(rng.uniform(-14, 14)))
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def clean_frame(rng: np.random.Generator, size: int = 416) -> np.ndarray:
    """A frame of undamaged belt: dark, slightly uneven, grainy. Synthetic."""
    base = rng.normal(55, 6, (size // 8, size // 8)).astype(np.float32)
    frame = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC) + rng.normal(0, 4, (size, size)).astype(np.float32)
    return cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--laps", type=int, default=7, help="laps to replay after the calibration lap (default %(default)s)")
    parser.add_argument("--damages", type=int, default=7, help="dataset images to place on the belt (default %(default)s)")
    parser.add_argument("--clean-frames", type=int, default=6, help="clean frames per lap (default %(default)s)")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--backend-url", default=settings.backend_url)
    parser.add_argument("--keep", action="store_true", help="do not empty the damage registry first")
    args = parser.parse_args()
    api = args.backend_url.rstrip("/")
    rng = np.random.default_rng(args.seed)

    if not TEST_IMAGES.is_dir():
        print("Dataset not found. Run:  python scripts/download_dataset.py", file=sys.stderr)
        return 1
    try:
        health = requests.get(f"{api}/health", timeout=5)
        health.raise_for_status()
    except requests.RequestException as exc:
        print(f"Backend not reachable at {api} ({type(exc).__name__}). Start it with:  uvicorn backend.main:app", file=sys.stderr)
        return 1
    if not args.keep:
        print("Emptying the damage registry:", requests.post(f"{api}/api/vision/reset", timeout=30).json())

    files = sorted(TEST_IMAGES.glob("*.jpg"))
    chosen = [files[i] for i in sorted(rng.choice(len(files), args.damages, replace=False))]
    loop = settings.belt_loop_length_m
    speed = settings.nominal_belt_speed_mps
    positions = sorted(float(p) for p in rng.uniform(200, loop - 200, args.damages - 1))
    positions.insert(1, positions[0] + 3.0)  # two damages 3 m apart
    damages = [PlacedDamage(f"D{i + 1} ({path.stem[:18]})", cv2.imread(str(path)), pos) for i, (path, pos) in enumerate(zip(chosen, positions))]
    damages[2].grows = True
    damages[3].last_lap = 2
    damages[4].first_lap = 3
    print(f"\nBelt loop {loop:g} m at {speed:g} m/s: one lap takes {loop / speed / 60:.1f} min. Placed {len(damages)} damages:")
    for d in damages:
        notes = [n for n, on in (("grows", d.grows), ("repaired after lap 2", d.last_lap == 2), ("appears on lap 3", d.first_lap == 3)) if on]
        print(f"  {d.name:<26} at {d.position_m:8.1f} m" + (f"   [{', '.join(notes)}]" if notes else ""))

    # --- run the belt --------------------------------------------------------------------
    lap_seconds = loop / speed
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=(args.laps + 1) * lap_seconds)
    tracker = BeltTracker(loop)
    tracker.marker_seen()  # the run starts with the marker at the camera

    # Lap 0 is a calibration lap: nothing is sent; the tracker learns its speed error from marker to marker.
    tracker.advance(lap_seconds, speed * (1 + SPEED_BIAS))
    drift = tracker.marker_seen()
    print(f"\nCalibration lap: position estimate was off by {drift:.1f} m at the marker; speed scale corrected to {tracker.scale:.5f}.")

    sent = found = errors = 0
    worst_error = 0.0
    session = requests.Session()
    for lap in range(1, args.laps + 1):
        events = [(d.position_m, d) for d in damages if d.first_lap <= lap <= d.last_lap]
        events += [(float(p), None) for p in rng.uniform(50, loop - 50, args.clean_frames)]
        events.sort(key=lambda e: e[0])
        measured_speed = speed * (1 + SPEED_BIAS + rng.normal(0, SPEED_JITTER))
        at = 0.0
        lap_found = 0
        for true_position, damage in events:
            estimate = tracker.advance((true_position - at) / speed, measured_speed)
            at = true_position
            worst_error = max(worst_error, abs(estimate - true_position))
            frame = clean_frame(rng) if damage is None else vary(damage.image, rng, zoom=1.0 + 0.07 * (lap - 1) if damage.grows else 1.0)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            when = start + timedelta(seconds=lap * lap_seconds + true_position / speed)
            response = session.post(
                f"{api}/api/vision/detect",
                files={"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
                data={"belt_position_m": f"{estimate % loop:.2f}", "lap": str(tracker.lap), "captured_at": when.isoformat()},
                timeout=120,
            )
            sent += 1
            if response.status_code == 503:
                print(response.json()["detail"], file=sys.stderr)
                return 1
            if response.status_code != 200:
                errors += 1
                print(f"  frame at {true_position:.0f} m refused: HTTP {response.status_code} {response.text[:120]}")
                continue
            lap_found += len(response.json()["detections"])
        tracker.advance((loop - at) / speed, measured_speed)
        drift = tracker.marker_seen()
        found += lap_found
        print(f"Lap {lap}: {len(events)} frames sent, {lap_found} detections; drift at the marker {drift:+.1f} m")

    # --- what the registry made of it ------------------------------------------------------
    registry = requests.get(f"{api}/api/vision/damages", timeout=30).json()
    print(f"\nFrames sent: {sent} ({errors} refused). Largest position error during the run: {worst_error:.1f} m.")
    print(f"Raw detections: {found}.  Damages placed on the belt: {len(damages)}.  Unique damages in the registry: {len(registry)}.")
    print("\nRegistry (by position along the belt loop):")
    print(f"  {'id':>3} {'position m':>10} {'status':<16} {'laps seen':>9} {'first-last lap':>14} {'growth':<14} {'%/lap':>6}  nearest placed damage")
    for entry in registry:
        nearest = min(damages, key=lambda d: min(abs(d.position_m - entry["belt_position_m"]), loop - abs(d.position_m - entry["belt_position_m"])))
        gap = abs(nearest.position_m - entry["belt_position_m"])
        pct = "" if entry["growth_pct_per_lap"] is None else f"{entry['growth_pct_per_lap']:+.1f}"
        print(f"  {entry['id']:>3} {entry['belt_position_m']:>10.1f} {entry['status']:<16} {entry['laps_seen']:>9} "
              f"{str(entry['first_lap']) + '-' + str(entry['last_lap']):>14} {entry['growth_trend']:<14} {pct:>6}  {nearest.name} ({gap:.1f} m away)")
    print("\nA naive count of detections would report", found, "damages. Open the dashboard's Vision tab to see the unique ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
