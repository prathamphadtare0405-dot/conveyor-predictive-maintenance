# Conveyor belt predictive maintenance

## 1. What this is

A software-only prototype that monitors a multi-kilometre mining conveyor and warns of
failures before they happen. A simulator plays the part of the sensor stations, including
faults that develop over hours and deliberately messy data. A backend stores the readings,
detects anomalies and predicts what might fail, where, and when. A vision component finds
damage in belt images and keeps a registry of unique damages on the moving belt loop. A
dashboard shows all of it.

Everything runs on a Windows laptop without a GPU. All numbers in this README were
produced by the code in this repository; the commands that produce them are given.

Two things to know before reading further:

- **All sensor results are on simulated data.** They show the pipeline works and where it
  is weak. They do not show how it would perform on a real conveyor.
- **The vision model is a stand-in.** It is trained on public photographs of cracks in
  roads and walls, because no belt-damage dataset could be verified and downloaded without
  an account. It is a prototype crack detector, not a belt damage detector (section 5).

Design document: [docs/DESIGN.md](docs/DESIGN.md). Decision log: [docs/DECISIONS.md](docs/DECISIONS.md).

## 2. Quick start (Windows)

Needs Python 3.11 or newer (built and run on 3.13.9) and about 3 GB of disk space. No
account and no API key are needed for anything.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_env.py
```

If PowerShell refuses `.venv\Scripts\activate` with "running scripts is disabled on this
system", either run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first
(it applies to that window only), use Command Prompt instead, or skip activation and
replace `python` with `.venv\Scripts\python.exe` in every command below.

`check_env.py` exercises every compiled library once. If it reports a failure with
"An Application Control policy has blocked this file", see the note at the end of this section.

Sensor side: generate history, train, evaluate.

```
python -m simulator.sensor_simulator --backfill 48
python -m ml.train
python -m ml.evaluation
```

Vision side: download the dataset (92 MB), prepare it, train (about 10 minutes on CPU).

```
python scripts/download_dataset.py
python -m vision.dataset
python -m vision.train
```

Run it. Either everything at once:

```
python run.py
```

or each part in its own terminal:

```
uvicorn backend.main:app --reload
python -m simulator.sensor_simulator --speedup 60 --inject bearing_wear:S04:3
streamlit run dashboard/app.py
```

Then, with the backend running, replay the belt loop for the vision tab:

```
python scripts/seed_demo.py
```

- Dashboard: http://localhost:8501
- API documentation: http://127.0.0.1:8000/docs
- Tests: `pytest -v` (110 tests, no GPU, no network, the YOLO model is mocked)

What to expect in the demo: `--inject bearing_wear:S04:3` starts a bearing fault at
station S04 that fails after 3 simulated hours. At 60x that is 3 to 5 minutes of real time.
S04 turns "abnormal", then "predicted failure", and the Predictions tab shows a time window
that narrows as the failure approaches.

Model weights and data are not committed; the commands above recreate them. Settings are
environment variables with defaults; copy `.env.example` to `.env` to change any.

**Note on Windows Smart App Control.** On the development laptop this Windows feature
blocked compiled files in the newest releases of PyTorch, scikit-learn, SQLAlchemy, pyarrow,
polars and fonttools. `requirements.txt` pins older releases of those that load. The
security setting was left on.

## 3. Approach

**Simulator** (`simulator/`). Ten stations at fixed kilometre positions report temperature,
vibration, belt speed, load and motor current every 5 seconds. Load is fed at the tail and
reaches each station after its travel time; current and temperature follow load;
temperature follows a daily cycle. Three failure modes develop gradually (bearing wear,
belt slip, overload) and two happen suddenly (blockage, spike). Every reading carries hidden
ground-truth columns in the CSV only: the fault mode, its stage, and the true time until
failure. These never reach the backend and exist so the evaluation is honest.

**Backend** (`backend/`). FastAPI and SQLite. Each incoming reading is checked on its own
and ends as accepted, flagged, rejected or duplicate, with the reason logged (section 7).

**Anomaly detection** (`ml/preprocessing.py`, `ml/predict.py`). Missing values are imputed
in one documented way. A baseline fitted on healthy data removes what load and time of day
explain, so the features describe what a fault changes. Rolling features over 30 minutes
(mean, spread, slope) feed an Isolation Forest trained on healthy data. A reading is
anomalous if the forest says so, if a signal is beyond its warning limit, or if a signal's
load-compensated level has drifted outside its healthy band.

**Prediction** (`ml/risk.py`). Rules and straight-line fits, no learned model. *What*: the
failure mode whose primary signal has moved furthest towards its limit. *Where*: the station
and its kilometre. *When*: the recent trend of that signal extended to the failure limit,
given as a window with a confidence level, or "no estimate" when there is no clear trend.
Every prediction stores a sentence saying which numbers it rests on.

**Vision** (`vision/`). Images are denoised and contrast-equalised, then a YOLO nano model
fine-tuned on CPU finds damage. Each detection is given a position along the belt loop and
a lap number and matched against a registry of known damages (section 6).

**Dashboard** (`dashboard/app.py`). Streamlit, reading only from the API: stations on a line
by kilometre, active anomalies, predictions with reasons, unique belt damages with growth,
per-sensor trends and a data-quality panel.

Analysis runs synchronously when readings arrive. That is fine for ten stations. At real
scale ingest would write to a queue, analysis would run in separate workers, and readings
would go to a time-series database.

## 4. Design decisions

Each with the alternative considered and why it lost. The full log, with 61 entries, is in
[docs/DECISIONS.md](docs/DECISIONS.md).

| Decision | Alternative | Why the alternative lost |
|---|---|---|
| Incoming readings are checked item by item by hand | A strict request model (Pydantic) | One NaN would reject the whole batch, including its good readings |
| Bad values are stored as NULL; gaps are filled only inside the analysis code | Impute at ingest and store the filled value | A stored imputed value cannot be told apart from a measurement later |
| Duplicates: the first reading wins | Last write wins | A late duplicate could change history that analysis already used |
| Two clocks: reading time for history and analysis, arrival time for "offline" | One clock | Late readings and time acceleration make them differ |
| Train the forest on healthy readings only | Train on everything, unlabelled | About 20 % of the data is fault data, too much to be treated as rare |
| A load and time-of-day baseline; trends are fitted on what is left | Trends on raw signals | Temperature rises every morning and with every load swing, which reads as a fault |
| Three anomaly rules (forest, limit, drift) | The Isolation Forest alone | The forest does not react to one signal leaving its training range (section 9) |
| Prediction by trend extrapolation with stated rules | A regressor trained on the simulator's time-to-failure labels | It would learn the simulator, and its numbers could not be checked by hand |
| A time window, and "no estimate" without a clear trend | A single timestamp, always | A point estimate hides the uncertainty, and a forced estimate is invented |
| Damage is located in belt-loop metres, with no link to a station | Attach damage to the nearest sensor station | The belt moves; the damage is at a different station every few minutes |
| Appearance check by template comparison | Perceptual hash of the detection box | The hash accepted 46 % of true repeat sightings, because the detector's boxes are not repeatable |
| One chart per signal | All signals on one chart with two axes | Different units; a second axis invites reading crossings that mean nothing |

## 5. Vision dataset

**Used:** crack-seg, published by Ultralytics
([documentation](https://docs.ultralytics.com/datasets/segment/crack-seg/)).

| | |
|---|---|
| Content | 4,029 photographs of cracks in roads and walls |
| Classes | one: `crack` |
| Annotations | instance-segmentation polygons, converted here to bounding boxes |
| Splits | 3,717 train, 200 validation, 112 test (counts checked after download) |
| Licence | marked Public Domain on its documentation page; cited source is the "crack" dataset on Roboflow Universe (university-bswxt/crack-bphdr, 2022) |
| Download | direct, from the URL in the dataset's own configuration file; no account |

**Why it was chosen.** The constraint was that everything must be free and need no account.
This was the only candidate whose facts could be verified and which downloads without one.
Cracks are also one of the damage types a belt cover shows.

**What that costs.** Concrete and asphalt are not rubber. The model has never seen a belt.
Its validation numbers (section 9) describe crack photographs, and nothing here claims it
would detect belt damage. This is a prototype of the pipeline, with a stand-in detector.

**What was rejected, and why:**

| Candidate | Why not |
|---|---|
| Roboflow Universe "conveyor belt tear", "CBDD" and similar | Pages refused automated access, so classes, size and licence could not be confirmed; download needs an account and API key |
| BeltCrack (GitHub UESTC-nnLab/BeltCrack) | The best match: real belt images in sequences. The repository is Apache-2.0, but the data sits on ScienceDB, whose page could not be read automatically, so annotation format, size and the dataset's own licence stayed unconfirmed |
| Mendeley iron-ore conveyor dataset (CC BY 4.0) | Foreign objects on a lab belt, not belt damage |
| Kaggle | No belt-damage dataset found; needs an account and API key |

To switch to a belt dataset: put it in YOLO format under `data/datasets/`, point
`vision/dataset.py` at it and retrain. Nothing else depends on the dataset.

## 6. The continuous-loop problem

The brief says only that the belt is a continuous loop and the vision component has to
account for that. The full analysis is in [docs/DESIGN.md, section 8](docs/DESIGN.md).

**Consequences**

- **Two reference frames.** Sensor stations are fixed to the structure. Damage is on the
  belt, which moves, so it has no conveyor kilometre. It has a position along the belt loop
  (10,030 m here), measured from a reference marker.
- **Repeat sightings.** A fixed camera sees the same damage once per lap, 34 times a day
  at 4 m/s. Counting detections counts one tear 34 times.
- **Repetition is also useful.** Every lap is a new measurement: growth can be tracked, one
  sighting can be confirmed by a second, and damage that stops appearing can be noticed.
- **Position drifts.** Position comes from integrating belt speed. A 1 % speed error is
  100 m per lap. A reference marker (a known splice) seen once per lap resets it.
- **Same or new?** The same damage looks different each lap, and two defects can be close.

**Handled**

- Each detection gets a belt-loop position and a lap number.
- Matching needs all of: same class, position within a tolerance that grows with distance
  from the marker (5 to 10 m), similar position across the belt, and similar appearance.
- Two boxes in one frame are never merged. A second frame in the same lap adds a sighting,
  not a lap.
- A match updates the entry (last seen, laps seen, size history). No match creates one.
- A damage passed without being seen for 3 laps is marked "not re-observed", not deleted,
  and becomes active again if it returns.
- A damage is "confirmed" once seen on two laps.
- The position tracker corrects its speed scale at each marker.
- The dashboard shows unique damages and a size trend per damage, not raw detections.

**How it was exercised, stated openly.** The dataset is unrelated still images, not footage
of one belt. `scripts/seed_demo.py` simulates the loop: it places 7 test-split images at
fixed belt positions and replays them for 7 laps, changing brightness, position in the frame
and sharpness each lap, with clean frames in between, a speed sensor that reads 0.4 % high,
and a marker event once per lap. One damage is scripted to grow, one to be repaired, one to
appear late, and two sit 3 m apart.

Result of that replay: the position estimate stayed within 3.5 m. 45 detections became 10
registry entries. Two images contain two cracks that the detector boxes separately, so 9 is
the right answer; one damage was registered twice. The repaired damage ended as
"not re-observed" and the late one was first registered on lap 3.

**Not handled**

- Finding the reference splice in an image. The replay hands the marker event to the tracker.
- Real drift. The tolerance is a configured number, not one measured on a belt.
- Real appearance change: dust, water, lighting, wear. The appearance threshold was chosen
  on the same kind of replayed images it is demonstrated on.
- Growth in millimetres, which needs camera calibration. Box area from this detector is
  noisy: the growing damage was reported as growing, but not by a believable amount.
- Damages that merge or split, and damage wider than one frame.
- A repair that changes the loop length, which shifts every position after it.
- The other face of the belt, and anything inside the carcass.

## 7. How messy data is handled

The simulator produces eight kinds of bad data on purpose (3 % of readings by default).
Nothing is dropped silently: every reading ends in one of four outcomes, and all but
"accepted" leave a row in a quality log with the reason and the offending payload.

| Problem | What the backend does |
|---|---|
| Missing signal, or null | Stored; that signal is NULL; flagged `missing_field` |
| NaN or infinity | Stored; NULL; flagged `nan` |
| Impossible value (for example -999 °C) | Stored; NULL; flagged `out_of_range` |
| Text in a numeric field | Stored; NULL; flagged `wrong_type`. Plain numeric text ("45.2") is used and still flagged |
| Missing or unreadable timestamp, no sensor id, or no usable signal at all | Rejected with HTTP 422 and a reason: the reading cannot be placed |
| Duplicate (same sensor and timestamp) | Not stored again; the first one is kept; logged |
| Out-of-order reading | Stored at its own timestamp, marked `arrived_late`; history is always sorted by reading time |
| Stuck sensor | From the 6th identical value in a row, flagged `stuck` and ignored by the analysis |
| Silent station | Status `offline` after 60 s without a message |

A batch gets HTTP 200 with a per-item list of problems when anything in it was usable, and
422 when nothing was. `GET /api/data-quality` gives totals, counts per sensor and per
reason, and the latest log entries. In the analysis, a missing value takes the station's
last valid value if it is at most 1 minute old, otherwise the healthy median, so a NaN can
never reach the model.

The vision endpoint rejects empty files, non-images and corrupt images with 422, and
answers 503 with instructions when no model has been trained.

## 8. Built, simplified, skipped

| Part | State | Reason |
|---|---|---|
| Simulator with gradual faults, ground truth, 8 kinds of messy data | Built | Everything else depends on it |
| Ingest, validation, de-duplication, ordering, offline detection | Built | Core requirement |
| Isolation Forest with rolling features | Built | Required; found to be the weakest of the three rules |
| Limit and drift rules | Built (added) | The forest alone missed slow single-signal drift |
| What / where / when prediction with window and confidence | Built | Second priority after the live pipeline |
| Evaluation against ground truth | Built | The only way to say anything true about the predictor |
| Vision detection with YOLO nano on CPU | Built, on a stand-in dataset | No belt dataset was obtainable without an account |
| Damage registry on the belt loop | Built | Third priority; the part the brief asks to be thought through |
| Belt loop | Simulated by replaying still images | No multi-lap footage of one belt exists publicly |
| Reference splice detection | Skipped; the marker event is given | Needs footage of a real splice |
| Station layout | Simplified: evenly spaced, every station reports all five signals | The brief gives no layout; a real belt measures current only at drives |
| Ambient temperature | Simplified: hour of day stands in for it | The brief lists no ambient sensor |
| Analysis on ingest | Simplified: synchronous, in the request | Enough for ten stations; a queue and workers at real scale |
| Time acceleration | Limited: `--speedup 60` runs at about 36x | Synchronous analysis takes about 160 ms per request; nothing is lost |
| Database | Simplified: SQLite, no migrations | One process, one laptop |
| Authentication, user accounts | Skipped | Not asked for; `POST /api/vision/reset` would need it in a real deployment |
| Curved (accelerating) trend fits | Skipped | A straight line is checkable by hand; its cost is measured in section 9 |
| Faults that spread between stations, belt mistracking, gearbox faults | Skipped | Each needs a signal or a coupling model the brief does not include |
| Alerts by email or message | Skipped | The dashboard is the stated output |

## 9. Evaluation results and their limits

### Sensor side: `python -m ml.evaluation`

Full report: [docs/evaluation_results.md](docs/evaluation_results.md). The evaluation run
is 72 simulated hours of the same simulated conveyor as the training data on different
days (different load, noise, faults and messy data), 506,345 readings, 12 gradual faults.

**Anomaly detector**

| Metric | Value |
|---|---|
| Precision | 0.957 |
| Recall (all fault readings, including the early stage) | 0.920 |
| False-alarm rate on stations healthy for a full window | 0.0008 |
| False-alarm events | 13 in 72 hours (0.62 per station per day) |
| Recall by stage | early 74.7 %, developing 100 %, advanced 100 % |

Contribution of each rule, as the share of fault readings it flags: warning limit 0.689,
drift 0.900, Isolation Forest 0.280. **Where neither other rule fired, the forest flagged
0.2 % of fault readings.** A tree can only split inside the range of its training data, so
a value far outside that range scores the same as the most extreme healthy value. The
forest is kept because the brief asks for it and because it is the only rule that can
catch an unusual combination of signals that are each in range; on this data it adds
almost nothing.

**Failure predictor**

| Metric | Value |
|---|---|
| Faults with a High or Critical warning before failure | 12 of 12 |
| Warning lead time | median 14.0 h (shortest 3.2 h, longest 21.4 h) |
| Failure mode named correctly at the warning | 11 of 12 |
| Failure mode correct, over all predictions that named one | 99.9 % of 995 |
| True failure time inside the estimated window | 72.2 % of 760 windows |
| Failure came sooner than the window | 26.1 % |
| Failure came later than the window | 1.7 % |
| Median error of the window midpoint | 2.4 h |
| High or Critical warnings on healthy stations | 0 |

The weak point is the 26.1 %: when more than 4 hours remain, about a third of windows open
too late. Simulated wear accelerates, and a straight line through accelerating data
over-estimates the time left. The window is most reliable in the last 4 hours (87 % inside
between 1 and 4 hours before failure).

### Vision side: `python -m vision.train`

Full report: [docs/vision_training_results.md](docs/vision_training_results.md). YOLO11n,
800 training images, image size 320, 12 epochs, CPU, 10.4 minutes.

| Metric (200 validation images, class `crack`) | Value |
|---|---|
| Precision | 0.649 |
| Recall | 0.687 |
| mAP@0.5 | 0.663 |
| mAP@0.5:0.95 | 0.386 |

### Limits

- **Simulated data only.** One run of one seed; with 12 faults the percentages move a lot
  from run to run.
- The simulator drives each failing signal to exactly the limit the predictor extrapolates
  to. On a real belt the failure point is not known that precisely.
- Simulated faults change one or two signals smoothly. Real faults are noisier, stall and overlap.
- The drift rule and the evaluation setup were changed after seeing evaluation output,
  because they fixed defects (DECISIONS D31, D36). The predictor's numeric thresholds were
  set beforehand and not tuned afterwards.
- The vision numbers are for photographs of cracks in roads and walls, from the same source
  as the training images. They say nothing about belts.

## 10. Dashboard screenshots

Screenshots are taken by the author and placed in `screenshots/`.

| View | File |
|---|---|
| Belt status | `screenshots/belt_status.png` |
| Active anomalies | `screenshots/anomalies.png` |
| Predictions | `screenshots/predictions.png` |
| Vision: belt damage | `screenshots/vision.png` |
| Sensor trends | `screenshots/trends.png` |
| Data quality | `screenshots/data_quality.png` |

## 11. What I would do differently

- **Get real belt images first.** The vision half rests on a stand-in dataset and a
  simulated loop. A day spent obtaining BeltCrack, or filming any belt for ten laps, would
  have been worth more than anything built on top of the stand-in.
- **Fit accelerating wear.** A straight line is the reason a quarter of the windows open
  too late. A curve (exponential, or a line fitted to the logarithm of the distance to the
  limit) would be the next thing to try, measured with the same evaluation.
- **Question the Isolation Forest earlier.** Its weakness on single-signal drift appeared in
  a unit test, late. Looking at each rule's contribution on day one would have put the
  effort into the residual-based rules from the start.
- **Evaluate over many seeds.** One 72-hour run gives 12 faults. Twenty runs would give
  confidence intervals instead of single percentages.
- **Measure ambient temperature** in the simulator instead of using hour of day for it.
- **Keep recent readings in memory** per station instead of re-reading them from SQLite on
  every batch, which is what limits time acceleration to about 36x.
- **Match damages across a whole lap at once** instead of frame by frame. Deciding with all
  of a lap's detections in hand would avoid most of the double registrations.

## 12. How AI tools were used

The system was built with an AI coding assistant (Claude Code) working from a written
specification. What that means concretely:

- The specification given to the assistant is in [docs/SPEC.md](docs/SPEC.md), unchanged.
- [docs/DECISIONS.md](docs/DECISIONS.md) logs each significant decision with the options
  considered, including where the author's instruction overrode the assistant's
  recommendation (for example the dataset choice, D47) and where the assistant's first
  design was wrong and was replaced after testing (D31, D36, D59).
- The assistant was required to run what it built and report real output, to ask before
  the prediction and vision phases, and never to invent metrics, dataset facts or links.
  Where it could not verify something (dataset pages it could not open, the dashboard in a
  browser) the log says so.
- The chat history itself is in [ai_usage/](ai_usage/), added by the author.
