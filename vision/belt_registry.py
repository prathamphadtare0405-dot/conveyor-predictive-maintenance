"""Damage registry for a belt that is a continuous loop.

The reasoning behind this module is in docs/DESIGN.md, section 8. In short:

* Damage is on the belt, and the belt moves. Its position is metres along the belt loop
  from a reference marker, not a kilometre on the conveyor. Nothing here knows about
  sensor stations.
* A fixed camera sees the same damage once per lap. Counting detections would count one
  tear again every lap, so each detection is matched against what is already known.
* A match needs three things at once: the same class, a position within tolerance
  (along the belt and across it), and a similar appearance.
* An entry that stops being seen is marked "not re-observed" and kept.

This module holds the logic only and has no database or image dependencies beyond the
hash comparison, so it can be tested with plain objects. `backend/crud.py` stores the
entries; any object with the attributes of `DamageRecord` works with these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from backend.config import settings
from vision.preprocessing import hash_from_hex, hash_similarity

STATUS_ACTIVE = "active"
STATUS_NOT_REOBSERVED = "not re-observed"

GROWTH_MIN_LAPS = 3
# Per lap, relative to the first measured size. In the loop replay, damages that did not
# change at all showed trends of up to about +/-3 % per lap from box jitter alone, so
# anything below 5 % is reported as stable.
GROWTH_THRESHOLD_PCT = 5.0


@dataclass
class RegistryConfig:
    loop_length_m: float = field(default_factory=lambda: settings.belt_loop_length_m)
    # Tolerance along the belt = base + drift_per_m * (metres travelled since the marker).
    # Position error grows with distance from the last marker, so the tolerance does too.
    base_tolerance_m: float = field(default_factory=lambda: settings.registry_position_tolerance_m)
    drift_per_m: float = field(default_factory=lambda: settings.registry_drift_per_m)
    # Across the belt, as a share of the frame width. The belt wanders sideways a little.
    lateral_tolerance: float = field(default_factory=lambda: settings.registry_lateral_tolerance)
    # Appearance score (0..1) needed to call two sightings the same damage, and the function
    # that produces the score from (observation, entry). The backend uses the template
    # comparison in vision/appearance.py; the default here compares perceptual hashes, which
    # needs nothing but the two hash strings.
    min_similarity: float = field(default_factory=lambda: settings.registry_min_similarity)
    similarity: Callable[[Any, Any], float] = field(default=lambda obs, entry: hash_pair_similarity(obs, entry))
    missed_laps: int = field(default_factory=lambda: settings.registry_missed_laps)
    # A damage counts as confirmed once it has been seen on this many different laps.
    confirm_laps: int = 2

    def tolerance_m(self, position_m: float) -> float:
        return self.base_tolerance_m + self.drift_per_m * position_m


@dataclass
class Observation:
    """One detection, placed on the belt."""

    damage_class: str
    position_m: float  # along the loop, 0 <= position < loop length
    lap: int
    lateral: float  # 0..1 across the frame
    appearance_hash: str  # hex, from vision.preprocessing.difference_hash
    area_fraction: float
    confidence: float
    timestamp: datetime
    appearance: Any = None  # vision.appearance.Appearance when the template comparison is used


@dataclass
class DamageRecord:
    """One unique damage. The database model has the same attributes."""

    id: int
    damage_class: str
    position_m: float
    lateral: float
    appearance_hash: str
    first_seen_at: datetime
    last_seen_at: datetime
    first_lap: int
    last_lap: int
    laps_seen: int = 1
    sightings: int = 1
    status: str = STATUS_ACTIVE
    best_confidence: float = 0.0
    last_confidence: float = 0.0


def hash_pair_similarity(obs: Any, entry: Any) -> float:
    """Share of equal bits between the perceptual hashes of an observation and an entry."""
    return hash_similarity(hash_from_hex(obs.appearance_hash), hash_from_hex(entry.appearance_hash))


def is_confirmed(entry: Any, cfg: RegistryConfig | None = None) -> bool:
    """Seen on at least two different laps. One sighting may be dirt, a shadow or a detector error."""
    return entry.laps_seen >= (cfg.confirm_laps if cfg else 2)


def circular_distance(a: float, b: float, loop_length_m: float) -> float:
    """Shortest distance between two belt positions, going either way round the loop."""
    d = abs(a - b) % loop_length_m
    return min(d, loop_length_m - d)


def circular_blend(old: float, new: float, weight: float, loop_length_m: float) -> float:
    """Move `old` a share `weight` of the way towards `new` along the shorter arc."""
    delta = (new - old + loop_length_m / 2.0) % loop_length_m - loop_length_m / 2.0
    return (old + weight * delta) % loop_length_m


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------


def match_score(obs: Observation, entry: Any, cfg: RegistryConfig) -> float | None:
    """How well an observation fits a registry entry, or None if it cannot be the same damage."""
    if entry.damage_class != obs.damage_class:
        return None
    tolerance = cfg.tolerance_m(obs.position_m)
    distance = circular_distance(obs.position_m, entry.position_m, cfg.loop_length_m)
    if distance > tolerance:
        return None
    if abs(obs.lateral - entry.lateral) > cfg.lateral_tolerance:
        return None
    similarity = cfg.similarity(obs, entry)
    if similarity < cfg.min_similarity:
        return None
    # Appearance decides between candidates; distance breaks ties.
    return similarity - 0.2 * (distance / tolerance)


def assign(observations: Sequence[Observation], entries: Sequence[Any], cfg: RegistryConfig) -> list[Any | None]:
    """Match the detections of ONE frame to registry entries, one to one.

    Two boxes in the same image are two different things, so an entry can take at most one
    of them. Pairs are taken best score first. Returns, for each observation, the matched
    entry or None (None means: this is a damage not seen before).
    """
    pairs = []
    for i, obs in enumerate(observations):
        for entry in entries:
            score = match_score(obs, entry, cfg)
            if score is not None:
                pairs.append((score, i, entry))
    pairs.sort(key=lambda p: -p[0])
    result: list[Any | None] = [None] * len(observations)
    taken: set[int] = set()
    for _score, i, entry in pairs:
        if result[i] is None and id(entry) not in taken:
            result[i] = entry
            taken.add(id(entry))
    return result


def apply_sighting(entry: Any, obs: Observation, cfg: RegistryConfig) -> None:
    """Update an entry with a new sighting of the same damage."""
    if obs.lap != entry.last_lap:
        entry.laps_seen += 1  # a second frame of the same pass does not count as another lap
    entry.sightings += 1
    entry.last_lap = max(entry.last_lap, obs.lap)
    entry.last_seen_at = max(entry.last_seen_at, obs.timestamp)
    # Follow slow changes: position drifts a little, and the damage itself changes.
    entry.position_m = circular_blend(entry.position_m, obs.position_m, 0.3, cfg.loop_length_m)
    entry.lateral = 0.7 * entry.lateral + 0.3 * obs.lateral
    entry.appearance_hash = obs.appearance_hash  # compare the next lap with the latest look
    entry.last_confidence = obs.confidence
    entry.best_confidence = max(entry.best_confidence, obs.confidence)
    entry.status = STATUS_ACTIVE  # a damage that was not re-observed and comes back is active again


def passes_missed(entry: Any, lap: int, position_m: float, cfg: RegistryConfig) -> int:
    """How many times the camera has gone past this damage's place without seeing it.

    In the current lap the place only counts as passed once the camera is beyond it by
    more than the tolerance.
    """
    laps_since = lap - entry.last_lap
    if laps_since <= 0:
        return 0
    passed_this_lap = position_m > entry.position_m + cfg.tolerance_m(entry.position_m)
    return laps_since - (0 if passed_this_lap else 1)


def update_statuses(entries: Iterable[Any], lap: int, position_m: float, cfg: RegistryConfig) -> list[Any]:
    """Mark active entries missed for `missed_laps` passes as "not re-observed". Nothing is deleted."""
    changed = []
    for entry in entries:
        if entry.status == STATUS_ACTIVE and passes_missed(entry, lap, position_m, cfg) >= cfg.missed_laps:
            entry.status = STATUS_NOT_REOBSERVED
            changed.append(entry)
    return changed


# --------------------------------------------------------------------------------------
# Growth
# --------------------------------------------------------------------------------------


def growth_trend(size_by_lap: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Trend of a damage's size over laps.

    `size_by_lap` is (lap, size) with one value per lap. Size is the detection box area as
    a share of the frame, so growth is relative, not in millimetres. A straight line is
    fitted over laps; its slope is reported in percent of the first size per lap. Fewer
    than three laps give no trend: two points always make a line.
    """
    points = sorted(size_by_lap)
    if len(points) < GROWTH_MIN_LAPS or points[0][1] <= 0:
        return {"trend": "too few laps", "growth_pct_per_lap": None, "laps_measured": len(points)}
    laps = np.array([p[0] for p in points], dtype=float)
    sizes = np.array([p[1] for p in points], dtype=float)
    slope = float(np.polyfit(laps, sizes, 1)[0])
    pct = 100.0 * slope / float(sizes[0])
    trend = "growing" if pct > GROWTH_THRESHOLD_PCT else "shrinking" if pct < -GROWTH_THRESHOLD_PCT else "stable"
    return {"trend": trend, "growth_pct_per_lap": round(pct, 1), "laps_measured": len(points)}


# --------------------------------------------------------------------------------------
# Belt position
# --------------------------------------------------------------------------------------


class BeltTracker:
    """Belt position at the camera by dead reckoning, corrected by the reference marker.

    `advance` integrates measured speed over time. The measurement is never exact, so the
    estimate drifts. Each time the marker (the reference splice) is seen, the position is
    set to zero, the lap number goes up, and the distance the tracker *thought* it had
    travelled since the previous marker is compared with the known loop length to correct
    the speed scale for the next lap.
    """

    def __init__(self, loop_length_m: float | None = None) -> None:
        self.loop_length_m = loop_length_m or settings.belt_loop_length_m
        self.position_m = 0.0
        self.lap = 0
        self.scale = 1.0  # correction applied to measured speed
        self._since_marker = 0.0
        self._marker_seen = False

    def advance(self, seconds: float, measured_speed_mps: float) -> float:
        """Move on by `seconds` at the measured speed. Returns the position within the loop."""
        distance = seconds * measured_speed_mps * self.scale  # signed: a reversing belt goes backwards
        self._since_marker += distance
        self.position_m += distance
        # With a marker, the lap changes when the marker is seen. Without one (not seen yet,
        # or missed for half a lap beyond where it was due), laps are counted by distance.
        limit = self.loop_length_m * (1.5 if self._marker_seen else 1.0)
        while self.position_m >= limit:
            self.position_m -= self.loop_length_m
            self.lap += 1
        while self.position_m < 0.0:
            self.position_m += self.loop_length_m
            self.lap -= 1
        return self.loop_position()

    def marker_seen(self) -> float:
        """Call when the reference marker passes the camera.

        Returns the drift that had built up since the previous marker, in metres (positive:
        the tracker had over-estimated the distance). The first marker has nothing to
        compare with and returns 0.
        """
        drift = 0.0
        travelled = self._since_marker
        if self._marker_seen and travelled > 0.5 * self.loop_length_m:
            # The belt really travelled one loop; the tracker measured `travelled`.
            drift = travelled - self.loop_length_m
            self.scale *= self.loop_length_m / travelled
        if travelled > 0.0:
            self.lap += 1
        self._marker_seen = True
        self.position_m = 0.0
        self._since_marker = 0.0
        return drift

    def loop_position(self) -> float:
        """Position folded into 0..loop length (the raw estimate can run past the end before the marker shows)."""
        return self.position_m % self.loop_length_m


# --------------------------------------------------------------------------------------
# In-memory registry (used by tests; the backend keeps the entries in the database)
# --------------------------------------------------------------------------------------


class DamageRegistry:
    def __init__(self, cfg: RegistryConfig | None = None) -> None:
        self.cfg = cfg or RegistryConfig()
        self.entries: list[DamageRecord] = []

    def register_frame(self, observations: Sequence[Observation], lap: int, position_m: float) -> list[tuple[DamageRecord, bool]]:
        """Register the detections of one frame. Returns (entry, is_new) per observation."""
        matches = assign(observations, self.entries, self.cfg)
        out = []
        for obs, entry in zip(observations, matches):
            if entry is None:
                entry = DamageRecord(
                    id=len(self.entries) + 1, damage_class=obs.damage_class, position_m=obs.position_m, lateral=obs.lateral,
                    appearance_hash=obs.appearance_hash, first_seen_at=obs.timestamp, last_seen_at=obs.timestamp,
                    first_lap=obs.lap, last_lap=obs.lap, best_confidence=obs.confidence, last_confidence=obs.confidence,
                )
                self.entries.append(entry)
                out.append((entry, True))
            else:
                apply_sighting(entry, obs, self.cfg)
                out.append((entry, False))
        update_statuses(self.entries, lap, position_m, self.cfg)
        return out

    def active(self) -> list[DamageRecord]:
        return [e for e in self.entries if e.status == STATUS_ACTIVE]
