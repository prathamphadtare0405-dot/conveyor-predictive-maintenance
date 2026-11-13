"""Image preprocessing, the belt-loop damage registry, and the vision API (YOLO model mocked)."""

from __future__ import annotations

from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from backend.config import settings
from vision import belt_registry as reg
from vision import detection
from vision.dataset import convert_labels, polygon_line_to_box
from vision.preprocessing import (
    InvalidImage,
    decode_image,
    difference_hash,
    enhance,
    hash_from_hex,
    hash_similarity,
    hash_to_hex,
)

T0 = datetime(2026, 11, 13, 8, 0, 0)
LOOP = 10_030.0
LAP_TIME = timedelta(seconds=LOOP / 4.0)


def textured(seed: int, size: int = 160) -> np.ndarray:
    """A grey image with coarse structure, so that hashes of different seeds differ."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(40, 200, (8, 8), dtype=np.uint8)
    image = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


# --- preprocessing -----------------------------------------------------------------------


def test_enhance_keeps_size_and_raises_contrast_of_a_dull_image():
    rng = np.random.default_rng(0)
    dull = np.clip(60 + rng.normal(0, 3, (120, 160, 3)), 0, 255).astype(np.uint8)
    dull[40:44, 20:140] = 50  # a faint dark line, like a crack under dust
    out = enhance(dull)
    assert out.shape == dull.shape and out.dtype == np.uint8
    line_before = float(dull[30:36].mean()) - float(dull[40:44, 20:140].mean())
    line_after = float(out[30:36].mean()) - float(out[40:44, 20:140].mean())
    assert line_after > 1.5 * line_before  # the line stands out more from its surroundings
    assert enhance(dull[:, :, 0]).shape == dull.shape  # a grey image is accepted too


def test_decode_rejects_what_is_not_an_image():
    assert decode_image(png(textured(1))).shape == (160, 160, 3)
    for bad, word in ((b"", "empty"), (b"this is not an image", "not a readable image"), (png(textured(1))[:60], "not a readable image")):
        with pytest.raises(InvalidImage, match=word):
            decode_image(bad)
    with pytest.raises(InvalidImage, match="too small"):
        decode_image(png(np.zeros((8, 8, 3), np.uint8)))
    big = decode_image(png(np.zeros((100, 3200, 3), np.uint8)))
    assert big.shape[1] == 1600  # oversized frames are scaled down


def test_hash_survives_brightness_blur_and_shift_but_separates_different_regions():
    image = textured(3)
    brighter = cv2.convertScaleAbs(image, alpha=1.15, beta=18)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    shifted = np.roll(image, 4, axis=1)
    base = difference_hash(image)
    assert hash_similarity(base, difference_hash(brighter)) > 0.9
    assert hash_similarity(base, difference_hash(blurred)) > 0.9
    assert hash_similarity(base, difference_hash(shifted)) > 0.8
    assert hash_similarity(base, difference_hash(textured(4))) < 0.62
    assert (hash_from_hex(hash_to_hex(base)) == base).all()


def test_polygon_labels_become_boxes():
    assert polygon_line_to_box("0 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7") == "0 0.400000 0.500000 0.400000 0.400000"
    assert polygon_line_to_box("0 0.5 0.5 0.2 0.1") == "0 0.500000 0.500000 0.200000 0.100000"  # already a box
    assert polygon_line_to_box("0 0.2 0.3 0.2 0.3 0.2 0.3") is None  # zero area
    assert polygon_line_to_box("crack 0.1 0.2") is None
    assert convert_labels("0 0.1 0.1 0.3 0.1 0.3 0.2\n\nbroken line\n") == "0 0.200000 0.150000 0.200000 0.100000\n"
    assert convert_labels("") == ""


# --- appearance (template comparison used by the backend) ---------------------------------


def scene(seed: int) -> np.ndarray:
    """A 320x320 grey scene with one dark crack-like polyline, different for each seed."""
    rng = np.random.default_rng(seed)
    image = np.clip(rng.normal(150, 6, (320, 320)), 0, 255).astype(np.uint8)
    points = np.column_stack([np.linspace(90, 230, 7) + rng.integers(-18, 18, 7), np.linspace(90, 230, 7)[rng.permutation(7)]]).astype(np.int32)
    cv2.polylines(image, [points], False, 40, 3)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_appearance_matches_the_same_damage_when_light_position_and_box_change():
    from vision import appearance

    image = scene(1)
    first = appearance.extract(image, 80, 80, 240, 240)
    later = cv2.convertScaleAbs(np.roll(image, (5, -4), axis=(0, 1)), alpha=1.12, beta=15)  # shifted and brighter
    assert appearance.similarity(first, appearance.extract(later, 80, 80, 240, 240)) > 0.8
    # The detector draws a different box next time: only part of the damage, or a looser box.
    assert appearance.similarity(first, appearance.extract(later, 100, 70, 200, 190)) > 0.55
    assert appearance.similarity(first, appearance.extract(later, 60, 60, 260, 260)) > 0.55
    # A different damage at the same place does not match.
    assert appearance.similarity(first, appearance.extract(scene(2), 80, 80, 240, 240)) < 0.55


def test_appearance_tolerates_moderate_growth_and_ignores_flat_patches():
    from vision import appearance

    image = scene(3)
    first = appearance.extract(image, 80, 80, 240, 240)
    grown = cv2.resize(image[24:296, 24:296], (320, 320))  # 18 % larger in the frame
    assert appearance.similarity(first, appearance.extract(grown, 70, 70, 250, 250)) > 0.55
    flat = np.full((320, 320, 3), 90, np.uint8)
    assert appearance.similarity(appearance.extract(flat, 80, 80, 240, 240), appearance.extract(flat, 80, 80, 240, 240)) == 0.0


def test_appearance_survives_storage_and_works_at_any_frame_size():
    from vision import appearance

    image = scene(4)
    look = appearance.extract(image, 80, 80, 240, 240)
    restored = appearance.Appearance.decode(look.encode(), look.x1, look.y1, look.x2, look.y2)
    assert (restored.context == look.context).all() and restored.patch.shape == look.patch.shape
    big = cv2.resize(image, (960, 960))  # the same scene from a higher-resolution camera
    assert appearance.similarity(look, appearance.extract(big, 240, 240, 720, 720)) > 0.8
    edge = appearance.extract(image, 0, 0, 30, 30)  # a box in the corner still yields a valid context
    assert edge.context.size > 0 and edge.patch.size > 0


# --- registry ----------------------------------------------------------------------------


def cfg(**overrides) -> reg.RegistryConfig:
    params = dict(loop_length_m=LOOP, base_tolerance_m=5.0, drift_per_m=0.0, lateral_tolerance=0.2, min_similarity=0.65, missed_laps=3)
    params.update(overrides)
    return reg.RegistryConfig(**params)


def look(seed: int, flip: int = 0) -> str:
    """An appearance hash; `flip` changes that many of its 256 bits (the same damage, seen a little differently)."""
    bits = np.random.default_rng(seed).random(256) < 0.5
    if flip:
        idx = np.random.default_rng(seed * 1000 + flip).choice(256, flip, replace=False)
        bits[idx] = ~bits[idx]
    return hash_to_hex(bits)


def obs(position: float, lap: int, seed: int = 1, flip: int = 0, lateral: float = 0.5, area: float = 0.05, cls: str = "crack") -> reg.Observation:
    return reg.Observation(cls, position, lap, lateral, look(seed, flip), area, 0.8, T0 + lap * LAP_TIME)


def test_same_damage_across_laps_is_counted_once():
    registry = reg.DamageRegistry(cfg())
    for lap in range(6):
        # Each lap: a slightly different position (drift) and a slightly different look.
        (entry, is_new), = registry.register_frame([obs(1200.0 + (-1) ** lap * 2.0, lap, flip=10 + lap)], lap, 1200.0)
        assert is_new == (lap == 0)
    assert len(registry.entries) == 1
    assert (entry.laps_seen, entry.sightings, entry.first_lap, entry.last_lap) == (6, 6, 0, 5)
    assert entry.position_m == pytest.approx(1200.0, abs=2.0) and entry.status == reg.STATUS_ACTIVE


def test_two_frames_of_the_same_pass_do_not_count_as_two_laps():
    registry = reg.DamageRegistry(cfg())
    registry.register_frame([obs(500.0, 0)], 0, 500.0)
    (entry, is_new), = registry.register_frame([obs(500.6, 0, flip=8)], 0, 500.6)  # overlapping frame
    assert not is_new and len(registry.entries) == 1
    assert entry.laps_seen == 1 and entry.sightings == 2


def test_two_nearby_damages_are_kept_separate():
    registry = reg.DamageRegistry(cfg())
    # 1.5 m apart along the belt, well inside the 5 m tolerance, but they look different.
    registry.register_frame([obs(800.0, 0, seed=1)], 0, 800.0)
    registry.register_frame([obs(801.5, 0, seed=2)], 0, 801.5)
    assert len(registry.entries) == 2
    for lap in (1, 2):
        (a, new_a), = registry.register_frame([obs(800.4, lap, seed=1, flip=12)], lap, 800.4)
        (b, new_b), = registry.register_frame([obs(801.2, lap, seed=2, flip=12)], lap, 801.2)
        assert not new_a and not new_b and a.id != b.id
    assert len(registry.entries) == 2 and [e.laps_seen for e in registry.entries] == [3, 3]


def test_two_boxes_in_one_frame_are_never_merged_even_if_they_look_alike():
    registry = reg.DamageRegistry(cfg(lateral_tolerance=1.0))
    first = registry.register_frame([obs(300.0, 0, seed=5, lateral=0.3), obs(300.0, 0, seed=5, flip=6, lateral=0.7)], 0, 300.0)
    assert [is_new for _, is_new in first] == [True, True]
    second = registry.register_frame([obs(300.5, 1, seed=5, flip=3, lateral=0.3), obs(300.5, 1, seed=5, flip=9, lateral=0.7)], 1, 300.5)
    assert [is_new for _, is_new in second] == [False, False]
    assert {e.id for e, _ in second} == {1, 2} and len(registry.entries) == 2


def test_same_place_and_look_but_other_side_of_the_belt_is_a_different_damage():
    registry = reg.DamageRegistry(cfg())
    registry.register_frame([obs(900.0, 0, lateral=0.15)], 0, 900.0)
    registry.register_frame([obs(900.0, 1, lateral=0.85)], 1, 900.0)
    assert len(registry.entries) == 2


def test_position_drift_within_tolerance_matches_and_beyond_it_does_not():
    registry = reg.DamageRegistry(cfg(base_tolerance_m=5.0))
    registry.register_frame([obs(2000.0, 0)], 0, 2000.0)
    assert registry.register_frame([obs(2004.5, 1, flip=5)], 1, 2004.5)[0][1] is False  # 4.5 m off: same damage
    assert registry.register_frame([obs(2012.0, 2, flip=5)], 2, 2012.0)[0][1] is True  # 9 m from the entry: a new one
    assert len(registry.entries) == 2


def test_tolerance_grows_with_distance_from_the_marker():
    c = cfg(base_tolerance_m=5.0, drift_per_m=0.001)
    assert c.tolerance_m(0.0) == 5.0 and c.tolerance_m(9000.0) == pytest.approx(14.0)
    registry = reg.DamageRegistry(c)
    registry.register_frame([obs(9000.0, 0)], 0, 9000.0)
    assert registry.register_frame([obs(9012.0, 1, flip=5)], 1, 9012.0)[0][1] is False  # 12 m off, but far from the marker
    near = reg.DamageRegistry(c)
    near.register_frame([obs(100.0, 0)], 0, 100.0)
    assert near.register_frame([obs(112.0, 1, flip=5)], 1, 112.0)[0][1] is True  # the same 12 m near the marker is too much


def test_positions_wrap_around_the_loop():
    assert reg.circular_distance(LOOP - 2.0, 1.5, LOOP) == pytest.approx(3.5)
    assert reg.circular_blend(LOOP - 1.0, 2.0, 0.5, LOOP) == pytest.approx(0.5)
    registry = reg.DamageRegistry(cfg())
    registry.register_frame([obs(LOOP - 2.0, 0)], 0, LOOP - 2.0)
    assert registry.register_frame([obs(1.5, 2, flip=5)], 2, 1.5)[0][1] is False
    assert len(registry.entries) == 1


def test_a_different_class_or_a_different_look_is_a_different_damage():
    registry = reg.DamageRegistry(cfg())
    registry.register_frame([obs(400.0, 0)], 0, 400.0)
    assert registry.register_frame([obs(400.0, 1, cls="tear")], 1, 400.0)[0][1] is True
    assert registry.register_frame([obs(400.0, 1, seed=99)], 1, 400.0)[0][1] is True
    assert len(registry.entries) == 3


def test_damage_not_seen_for_n_laps_is_marked_not_deleted_and_can_come_back():
    registry = reg.DamageRegistry(cfg(missed_laps=3))
    registry.register_frame([obs(1000.0, 0)], 0, 1000.0)
    entry = registry.entries[0]
    for lap in (1, 2):
        registry.register_frame([], lap, 9000.0)  # clean frames, camera well past the place
        assert entry.status == reg.STATUS_ACTIVE
    # Lap 3, camera not yet at the place: only two passes have been missed so far.
    registry.register_frame([], 3, 500.0)
    assert reg.passes_missed(entry, 3, 500.0, registry.cfg) == 2 and entry.status == reg.STATUS_ACTIVE
    registry.register_frame([], 3, 1010.0)  # now it has gone past for the third time
    assert entry.status == reg.STATUS_NOT_REOBSERVED
    assert len(registry.entries) == 1 and registry.active() == []
    (back, is_new), = registry.register_frame([obs(1001.0, 6, flip=10)], 6, 1001.0)
    assert not is_new and back is entry and entry.status == reg.STATUS_ACTIVE and entry.laps_seen == 2


def test_growth_trend_needs_three_laps_and_is_relative_to_the_first_size():
    assert reg.growth_trend([(0, 0.05), (1, 0.06)])["trend"] == "too few laps"
    growing = reg.growth_trend([(0, 0.050), (1, 0.055), (2, 0.060), (3, 0.065)])
    assert growing == {"trend": "growing", "growth_pct_per_lap": 10.0, "laps_measured": 4}
    assert reg.growth_trend([(0, 0.05), (1, 0.051), (2, 0.049), (3, 0.05)])["trend"] == "stable"
    assert reg.growth_trend([(0, 0.05), (2, 0.04), (4, 0.03)])["trend"] == "shrinking"
    assert reg.growth_trend([])["growth_pct_per_lap"] is None


def test_dead_reckoning_drifts_and_the_marker_removes_the_drift():
    lap_seconds = LOOP / 4.0
    uncorrected = reg.BeltTracker(LOOP)
    for _ in range(3):
        uncorrected.advance(lap_seconds, 4.0 * 1.01)  # speed sensor reads 1 % high
    assert uncorrected.lap == 3 and uncorrected.position_m == pytest.approx(3 * 0.01 * LOOP, rel=1e-6)  # 301 m of error

    tracker = reg.BeltTracker(LOOP)
    assert tracker.marker_seen() == 0.0 and tracker.lap == 0  # start of the run, marker at the camera
    tracker.advance(lap_seconds, 4.0 * 1.01)
    assert tracker.loop_position() == pytest.approx(0.01 * LOOP, rel=1e-6)  # 100 m off after one lap
    assert tracker.marker_seen() == pytest.approx(0.01 * LOOP, rel=1e-6)
    assert tracker.lap == 1 and tracker.position_m == 0.0
    tracker.advance(lap_seconds / 2.0, 4.0 * 1.01)
    assert tracker.position_m == pytest.approx(LOOP / 2.0, rel=1e-6)  # the scale was corrected
    tracker.advance(lap_seconds / 2.0, 4.0 * 1.01)
    assert abs(tracker.marker_seen()) < 1e-6 and tracker.lap == 2


# --- API, with the YOLO model replaced ---------------------------------------------------


class FakeBoxes:
    def __init__(self, boxes):
        self.xyxy = np.array([b[:4] for b in boxes], dtype=float).reshape(-1, 4)
        self.conf = np.array([b[4] for b in boxes], dtype=float)
        self.cls = np.array([0.0] * len(boxes))


class FakeResult:
    names = {0: "crack"}

    def __init__(self, boxes):
        self.boxes = FakeBoxes(boxes)


class FakeModel:
    """Stands in for YOLO: finds one 'crack' in any image that is not almost black."""

    calls = 0

    def predict(self, image, **kwargs):
        FakeModel.calls += 1
        assert kwargs["device"] == "cpu"
        boxes = [(30.0, 40.0, 110.0, 100.0, 0.83)] if image.mean() > 20 else []
        return [FakeResult(boxes)]


@pytest.fixture()
def vision_client(client, monkeypatch):
    settings.vision_weights.parent.mkdir(parents=True, exist_ok=True)
    settings.vision_weights.write_bytes(b"placeholder: the model itself is mocked")
    monkeypatch.setattr(detection, "_load_model", lambda weights: FakeModel())
    monkeypatch.setattr(detection, "_detector", None)
    yield client
    settings.vision_weights.unlink(missing_ok=True)
    monkeypatch.setattr(detection, "_detector", None)


def upload(client, image_bytes, name="frame.png", content_type="image/png", **form):
    return client.post("/api/vision/detect", files={"file": (name, image_bytes, content_type)}, data={k: str(v) for k, v in form.items()})


def test_missing_weights_give_a_clear_503_not_a_crash(client, monkeypatch):
    monkeypatch.setattr(detection, "_detector", None)
    r = upload(client, png(textured(1)))
    assert r.status_code == 503 and "python -m vision.train" in r.json()["detail"]
    assert client.get("/api/vision/damages").json() == []
    assert client.get("/health").status_code == 200  # the rest of the backend is unaffected


def test_detect_returns_boxes_class_confidence_and_saves_an_annotated_image(vision_client):
    r = upload(vision_client, png(textured(1)), belt_position_m=1234.5, lap=0, captured_at="2026-11-13T08:00:00Z")
    assert r.status_code == 200
    body = r.json()
    assert (body["belt_position_m"], body["lap"], body["position_source"]) == (1234.5, 0, "client")
    assert (body["image_width"], body["image_height"]) == (160, 160)
    det, = body["detections"]
    assert det["damage_class"] == "crack" and det["confidence"] == 0.83
    assert det["box"] == {"x1": 30.0, "y1": 40.0, "x2": 110.0, "y2": 100.0}
    assert det["is_new_damage"] is True and det["area_fraction"] == pytest.approx(80 * 60 / 160**2, abs=1e-4)
    image = vision_client.get(body["annotated_image_url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert decode_image(image.content).shape == (160, 160, 3)


@pytest.mark.parametrize(
    "data, content_type, word",
    [
        (b"sensor_id,temperature\nS01,52.0\n", "text/csv", "Expected an image"),
        (b"definitely not a jpeg", "image/jpeg", "not a readable image"),
        (b"", "image/png", "empty"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 40, "image/png", "not a readable image"),
    ],
)
def test_non_images_and_corrupt_files_are_rejected_cleanly(vision_client, data, content_type, word):
    r = upload(vision_client, data, content_type=content_type)
    assert r.status_code == 422 and word in r.json()["detail"]
    assert vision_client.get("/api/vision/detections").json()["total"] == 0


def test_position_fields_are_validated(vision_client):
    image = png(textured(1))
    assert upload(vision_client, image, belt_position_m=100).status_code == 422  # lap missing
    assert upload(vision_client, image, belt_position_m=20000, lap=0).status_code == 422  # beyond the loop
    assert upload(vision_client, image, belt_position_m=100, lap=-1).status_code == 422
    assert vision_client.post("/api/vision/detect").status_code == 422  # no file at all


def test_same_damage_seen_every_lap_is_one_damage_with_a_size_history(vision_client):
    start = datetime(2026, 11, 13, 8, 0, 0)
    for lap in range(4):
        frame = cv2.convertScaleAbs(textured(7), alpha=1.0 + 0.04 * lap, beta=5 * lap)  # lighting changes lap to lap
        when = (start + lap * LAP_TIME).isoformat() + "Z"
        body = upload(vision_client, png(frame), belt_position_m=4000.0 + lap * 0.8, lap=lap, captured_at=when).json()
        assert body["detections"][0]["is_new_damage"] == (lap == 0)
        # A clean (black) frame elsewhere on the belt: analysed, nothing found, nothing stored as damage.
        clean = upload(vision_client, png(np.zeros((160, 160, 3), np.uint8)), belt_position_m=7000.0, lap=lap, captured_at=when).json()
        assert clean["detections"] == [] and clean["annotated_image_url"] is None

    damages = vision_client.get("/api/vision/damages").json()
    assert len(damages) == 1  # four detections, one damage
    d = damages[0]
    assert (d["laps_seen"], d["sightings"], d["first_lap"], d["last_lap"], d["status"]) == (4, 4, 0, 3, "active")
    assert d["confirmed"] is True
    assert d["belt_position_m"] == pytest.approx(4001.0, abs=1.5)
    assert [p["lap"] for p in d["size_history"]] == [0, 1, 2, 3] and d["growth_trend"] == "stable"
    assert d["annotated_image_url"].endswith(".jpg")
    assert "sensor_id" not in d and "station_km" not in d  # damage is not attached to a sensor station

    raw = vision_client.get("/api/vision/detections").json()
    assert raw["total"] == 4 and {item["damage_id"] for item in raw["items"]} == {d["id"]}
    assert vision_client.get("/api/vision/detections", params={"lap": 2}).json()["total"] == 1
    summary = vision_client.get("/api/dashboard/summary").json()["vision"]
    assert summary["frames_analysed"] == 8 and summary["raw_detections"] == 4 and summary["unique_damages"] == 1


def test_unseen_damage_becomes_not_re_observed_through_the_api(vision_client):
    upload(vision_client, png(textured(7)), belt_position_m=4000.0, lap=0)
    black = png(np.zeros((160, 160, 3), np.uint8))
    for lap in (1, 2, 3):
        upload(vision_client, black, belt_position_m=4000.0, lap=lap)
        upload(vision_client, black, belt_position_m=9000.0, lap=lap)
    d, = vision_client.get("/api/vision/damages").json()
    assert d["status"] == "not re-observed" and d["laps_seen"] == 1
    assert d["confirmed"] is False  # seen on one lap only
    assert vision_client.get("/api/vision/damages", params={"confirmed_only": True}).json() == []
    assert vision_client.get("/api/vision/damages", params={"status": "active"}).json() == []
    assert len(vision_client.get("/api/vision/damages", params={"status": "not re-observed"}).json()) == 1


def test_without_a_position_the_backend_dead_reckons_from_time(vision_client):
    start = datetime(2026, 11, 13, 8, 0, 0)
    first = upload(vision_client, png(textured(7)), captured_at=start.isoformat() + "Z").json()
    assert (first["belt_position_m"], first["lap"], first["position_source"]) == (0.0, 0, "dead_reckoning")
    later = upload(vision_client, png(textured(7)), captured_at=(start + timedelta(seconds=3000)).isoformat() + "Z").json()
    # 3000 s at 4 m/s = 12 000 m = one lap (10 030 m) plus 1 970 m.
    assert (later["belt_position_m"], later["lap"]) == (1970.0, 1)
    assert upload(vision_client, png(textured(7)), captured_at=(start - timedelta(seconds=5)).isoformat() + "Z").status_code == 422
