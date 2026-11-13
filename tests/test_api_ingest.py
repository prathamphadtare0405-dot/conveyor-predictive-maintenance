"""Ingest API: every kind of bad input, de-duplication, out-of-order data, offline status."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from backend import crud
from tests.conftest import reading

URL = "/api/sensors/readings"


def post_raw(client, payload):
    """Send JSON that may contain NaN, which the standard encoder of the test client refuses."""
    return client.post(URL, content=json.dumps(payload), headers={"Content-Type": "application/json"})


def stamp(i: int) -> str:
    return (datetime(2026, 11, 13, 8, 0, 0) + timedelta(seconds=5 * i)).isoformat() + "Z"


def history(client, sensor_id="S01", **params):
    return client.get(f"/api/sensors/{sensor_id}/history", params={"order": "asc", "limit": 1000, **params}).json()


# --- good input ------------------------------------------------------------------------


def test_single_reading_is_stored(client):
    r = client.post(URL, json=reading())
    assert r.status_code == 200
    assert r.json() == {
        "received": 1, "accepted": 1, "flagged": 0, "rejected": 0, "duplicates": 0, "late": 0,
        "problems": [], "problems_truncated": False,
    }
    item = history(client)["items"][0]
    assert item["temperature"] == 52.0 and item["flags"] == "" and item["timestamp"] == "2026-11-13T08:00:00.000Z"


def test_batch_registers_stations_by_kilometre(client):
    batch = [reading("S02", station_km=0.75), reading("S01"), reading("S03", station_km=1.25)]
    assert client.post(URL, json=batch).json()["accepted"] == 3
    sensors = client.get("/api/sensors").json()
    assert [(s["sensor_id"], s["station_km"], s["status"]) for s in sensors] == [
        ("S01", 0.25, "normal"), ("S02", 0.75, "normal"), ("S03", 1.25, "normal"),
    ]
    assert sensors[0]["latest_reading"]["current"] == 260.0


# --- partly usable input: stored and flagged ---------------------------------------------


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"current": None}, "missing_field"),
        ({"current": float("nan")}, "nan"),
        ({"current": float("inf")}, "nan"),
        ({"current": -999.0}, "out_of_range"),
        ({"current": 1.0e6}, "out_of_range"),
        ({"current": "ERR"}, "wrong_type"),
        ({"current": "260.0A"}, "wrong_type"),
        ({"current": ""}, "wrong_type"),
        ({"current": True}, "wrong_type"),
        ({"current": [260.0]}, "wrong_type"),
    ],
)
def test_one_bad_signal_is_blanked_and_the_rest_kept(client, change, reason):
    r = post_raw(client, reading(**change))
    assert r.status_code == 200
    body = r.json()
    assert (body["accepted"], body["flagged"], body["rejected"]) == (0, 1, 0)
    assert body["problems"][0]["issues"][0]["reason"] == reason
    assert body["problems"][0]["issues"][0]["field"] == "current"
    item = history(client)["items"][0]
    assert item["current"] is None and item["temperature"] == 52.0
    assert item["flags"] == f"{reason}:current"


def test_absent_signal_is_flagged(client):
    item = reading()
    del item["vibration"]
    body = client.post(URL, json=item).json()
    assert body["flagged"] == 1 and body["problems"][0]["issues"][0] == {
        "reason": "missing_field", "field": "vibration", "detail": "field is absent",
    }
    assert history(client)["items"][0]["vibration"] is None


def test_number_sent_as_text_is_used_but_logged(client):
    body = client.post(URL, json=reading(temperature="52.5")).json()
    assert body["flagged"] == 1 and body["problems"][0]["issues"][0]["reason"] == "wrong_type"
    assert history(client)["items"][0]["temperature"] == 52.5


def test_station_km_that_disagrees_with_the_registered_position_is_flagged(client):
    client.post(URL, json=reading(ts=stamp(0)))
    body = client.post(URL, json=reading(ts=stamp(1), station_km=3.0)).json()
    assert body["flagged"] == 1 and body["problems"][0]["issues"][0]["reason"] == "station_km_mismatch"
    assert client.get("/api/sensors/S01").json()["station_km"] == 0.25


def test_known_sensor_may_omit_station_km(client):
    client.post(URL, json=reading(ts=stamp(0)))
    item = reading(ts=stamp(1))
    del item["station_km"]
    assert client.post(URL, json=item).json()["accepted"] == 1


# --- unusable input: rejected with 422 ---------------------------------------------------


@pytest.mark.parametrize(
    "item, reason",
    [
        ({k: v for k, v in reading().items() if k != "timestamp"}, "missing_timestamp"),
        (reading(ts=None), "missing_timestamp"),
        (reading(ts="yesterday"), "bad_timestamp"),
        (reading(ts=1763020800), "bad_timestamp"),
        ({k: v for k, v in reading().items() if k != "sensor_id"}, "missing_sensor_id"),
        (reading(sensor_id=""), "missing_sensor_id"),
        (reading(sensor_id=17), "missing_sensor_id"),
        (reading(temperature="ERR", vibration=None, speed=-5, load=float("nan"), current="x"), "no_usable_signals"),
        ({k: v for k, v in reading("S77").items() if k != "station_km"}, "unknown_sensor"),
    ],
)
def test_unusable_reading_is_rejected_with_a_reason(client, item, reason):
    r = post_raw(client, item)
    assert r.status_code == 422
    body = r.json()
    assert (body["accepted"], body["flagged"], body["rejected"]) == (0, 0, 1)
    assert body["problems"][0]["outcome"] == "rejected"
    assert body["problems"][0]["issues"][0]["reason"] == reason
    quality = client.get("/api/data-quality").json()
    assert quality["totals"]["rejected"] == 1 and quality["by_reason"] == {reason: 1}
    assert quality["totals"]["readings_stored"] == 0


@pytest.mark.parametrize("payload", ["just text", 42, [], None])
def test_body_that_is_not_a_reading_or_list_is_422(client, payload):
    assert client.post(URL, json=payload).status_code == 422


def test_invalid_json_and_oversized_batch_are_422(client):
    r = client.post(URL, content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    too_many = [reading(ts=stamp(i)) for i in range(5001)]
    r = client.post(URL, json=too_many)
    assert r.status_code == 422 and "exceeds the limit" in r.json()["detail"]


def test_mixed_batch_keeps_the_good_and_reports_the_bad(client):
    batch = [
        reading(ts=stamp(0)),
        "not an object",
        reading(ts=stamp(1), load="N/A"),
        reading(ts="garbage"),
        reading(ts=stamp(0)),  # duplicate of the first
    ]
    r = client.post(URL, json=batch)
    assert r.status_code == 200
    body = r.json()
    assert {k: body[k] for k in ("received", "accepted", "flagged", "rejected", "duplicates")} == {
        "received": 5, "accepted": 1, "flagged": 1, "rejected": 2, "duplicates": 1,
    }
    assert [(p["index"], p["outcome"]) for p in body["problems"]] == [
        (1, "rejected"), (2, "flagged"), (3, "rejected"), (4, "duplicate"),
    ]
    assert history(client)["total"] == 2


# --- duplicates and ordering -------------------------------------------------------------


def test_duplicate_across_requests_is_not_stored_twice_and_first_value_wins(client):
    client.post(URL, json=reading(temperature=50.0))
    r = client.post(URL, json=reading(temperature=99.0))
    assert r.status_code == 200 and r.json()["duplicates"] == 1
    h = history(client)
    assert h["total"] == 1 and h["items"][0]["temperature"] == 50.0
    assert client.get("/api/data-quality").json()["per_sensor"][0]["duplicates"] == 1


def test_same_timestamp_on_different_sensors_is_not_a_duplicate(client):
    body = client.post(URL, json=[reading("S01"), reading("S02", station_km=0.75)]).json()
    assert body["accepted"] == 2 and body["duplicates"] == 0


def test_timestamps_with_offsets_are_compared_in_utc(client):
    client.post(URL, json=reading(ts="2026-11-13T08:00:00Z"))
    body = client.post(URL, json=reading(ts="2026-11-13T13:30:00+05:30")).json()
    assert body["duplicates"] == 1


def test_out_of_order_readings_are_placed_by_their_own_timestamp(client):
    client.post(URL, json=[reading(ts=stamp(0)), reading(ts=stamp(1)), reading(ts=stamp(4))])
    body = client.post(URL, json=[reading(ts=stamp(3), temperature=53.0), reading(ts=stamp(2), temperature=52.5)]).json()
    assert body["accepted"] == 2 and body["late"] == 2
    items = history(client)["items"]
    assert [i["timestamp"] for i in items] == [stamp(i).replace("Z", ".000Z") for i in range(5)]
    assert [i["arrived_late"] for i in items] == [False, False, True, True, False]
    # The newest reading is still the one with the newest timestamp, not the last to arrive.
    assert client.get("/api/sensors/S01").json()["latest_reading"]["timestamp"] == stamp(4).replace("Z", ".000Z")
    assert client.get("/api/data-quality").json()["totals"]["late"] == 2


def test_history_pagination_and_date_filter(client):
    client.post(URL, json=[reading(ts=stamp(i), load=float(i)) for i in range(10)])
    page = client.get("/api/sensors/S01/history", params={"limit": 3, "offset": 2}).json()
    assert page["total"] == 10 and [i["load"] for i in page["items"]] == [7.0, 6.0, 5.0]  # newest first by default
    window = history(client, start=stamp(4), end=stamp(6))
    assert [i["load"] for i in window["items"]] == [4.0, 5.0, 6.0]
    assert client.get("/api/sensors/S01/history", params={"start": stamp(6), "end": stamp(4)}).status_code == 422
    assert client.get("/api/sensors/S99/history").status_code == 404
    assert client.get("/api/sensors/S99").status_code == 404


# --- stuck sensor ------------------------------------------------------------------------


def test_signal_repeating_exactly_is_flagged_as_stuck(client):
    batch = [
        reading(ts=stamp(i), temperature=50.0 + i * 0.1, speed=4.0 + i * 0.001, load=70.0 + i, current=260.0 + i, vibration=2.777)
        for i in range(8)
    ]
    body = client.post(URL, json=batch).json()
    # Six identical values in a row are needed, so readings 6, 7 and 8 are flagged.
    assert body["accepted"] == 5 and body["flagged"] == 3
    assert all(p["issues"] == [{"reason": "stuck", "field": "vibration", "detail": "same value 2.777 for 6 or more readings in a row"}] for p in body["problems"])
    # It carries on across requests, and stops when the value moves again.
    assert client.post(URL, json=reading(ts=stamp(8), vibration=2.777)).json()["flagged"] == 1
    assert client.post(URL, json=reading(ts=stamp(9), vibration=2.801)).json()["accepted"] == 1


# --- offline -----------------------------------------------------------------------------


def test_station_goes_offline_when_nothing_arrives_and_recovers(client, monkeypatch):
    client.post(URL, json=[reading("S01", ts=stamp(0)), reading("S02", ts=stamp(0), station_km=0.75)])
    real_now = crud.utcnow()
    monkeypatch.setattr(crud, "utcnow", lambda: real_now + timedelta(seconds=61))
    client.post(URL, json=reading("S02", ts=stamp(1), station_km=0.75))  # only S02 keeps talking
    status = {s["sensor_id"]: s["status"] for s in client.get("/api/sensors").json()}
    assert status == {"S01": "offline", "S02": "normal"}
    assert client.get("/api/dashboard/summary").json()["status_counts"] == {"offline": 1, "normal": 1}
    client.post(URL, json=reading("S01", ts=stamp(2)))
    assert client.get("/api/sensors/S01").json()["status"] == "normal"


# --- reporting ---------------------------------------------------------------------------


def test_data_quality_counts_per_sensor(client):
    post_raw(client, [
        reading("S01", ts=stamp(0)),
        reading("S01", ts=stamp(1), current=float("nan")),
        reading("S01", ts=stamp(1)),  # duplicate
        reading("S02", ts=stamp(0), station_km=0.75, speed="ERR", load=-999),
        reading("S02", ts="bad", station_km=0.75),
    ])
    q = client.get("/api/data-quality").json()
    assert q["totals"] == {"readings_stored": 3, "flagged_readings": 2, "rejected": 1, "duplicates": 1, "late": 0}
    per = {s["sensor_id"]: s for s in q["per_sensor"]}
    assert per["S01"]["by_reason"] == {"nan": 1, "duplicate": 1}
    assert per["S02"]["by_reason"] == {"wrong_type": 1, "out_of_range": 1, "bad_timestamp": 1}
    assert per["S02"]["rejected"] == 1 and per["S02"]["flagged_readings"] == 1
    assert len(q["recent_events"]) == 5
    rejected = next(e for e in q["recent_events"] if e["outcome"] == "rejected")
    assert rejected["reason"] == "bad_timestamp" and '"timestamp": "bad"' in rejected["raw"]


def test_health_and_summary(client):
    assert client.get("/health").json()["status"] == "ok"
    empty = client.get("/api/dashboard/summary").json()
    assert empty["sensors_total"] == 0 and empty["latest_reading_time"] is None
    client.post(URL, json=reading())
    summary = client.get("/api/dashboard/summary").json()
    assert summary["sensors_total"] == 1 and summary["readings_stored"] == 1
    assert summary["latest_reading_time"] == "2026-11-13T08:00:00.000Z"
    assert client.get("/docs").status_code == 200
