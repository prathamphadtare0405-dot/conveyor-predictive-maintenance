# Project: Conveyor Belt Predictive Maintenance System (take-home assessment)

I have a take-home assessment for an AI Systems Engineer role. Build a working end-to-end
system in this workspace that monitors a multi-kilometre mining conveyor belt and predicts
failures before they happen. Everything runs in software on a Windows laptop with no GPU.

Save this whole prompt as `docs/SPEC.md` first, and re-read it at the start of every phase.

## 0. How the work is judged (read this before designing anything)

The assessors say, in their own brief:

- They would rather see a few parts done well, with clear reasoning about what was skipped,
  than every part done thinly.
- They read the README before the code, and judge judgement and writing as much as code.
- They look at: problem breakdown, handling of messy input, build-vs-skip decisions,
  clarity of writing, and how AI tools were used.
- Deadline: 17 Nov 2026, 11:59 PM IST. Submission is one GitHub repo.

So: depth over breadth. When you run short of time or hit a wall, cut scope and write down
what you cut and why in `docs/DECISIONS.md`. Never pad with thin features.

Priority order if something has to give:
1. Simulator -> backend -> prediction -> dashboard working together on live data
2. Prediction that answers "what might fail, where, and when"
3. Vision loop handling (section 6)
4. README and design document
5. Everything else (extra charts, extra endpoints, extra tests)

## 1. The five deliverables

| # | Component | What it must do |
|---|-----------|-----------------|
| 1 | Design document | Explain what is monitored, how, and why |
| 2 | Backend | Receive, store and analyse the data |
| 3 | Prediction model | Predict possible failures |
| 4 | Vision component | Detect belt wear and tear from images |
| 5 | Dashboard | Make the belt's health visible |

## 2. Stack

Python 3.11+, FastAPI, SQLite + SQLAlchemy, scikit-learn, Ultralytics YOLO (nano model) +
OpenCV, Streamlit + Plotly, pandas, NumPy, joblib, pytest. All config through environment
variables with a `.env.example`. If you want to change any of this, ask me first.

## 3. Project structure

```
conveyor-predictive-maintenance/
├── backend/        __init__.py, main.py, config.py, database.py, models.py,
│                   schemas.py, crud.py, routes/ (sensors, anomalies, predictions, vision, system)
├── simulator/      __init__.py, sensor_simulator.py, fault_models.py
├── ml/             __init__.py, preprocessing.py, train.py, predict.py, risk.py, evaluation.py, models/
├── vision/         __init__.py, dataset.py, preprocessing.py, detection.py, belt_registry.py, models/
├── dashboard/      app.py
├── scripts/        download_dataset.py, seed_demo.py
├── tests/
├── docs/           SPEC.md, DESIGN.md, DECISIONS.md
├── ai_usage/
├── screenshots/
├── data/           (generated at runtime, git-ignored except small test fixtures)
├── .env.example, .gitignore, requirements.txt, README.md, run.py
```

Do not commit the SQLite file, generated CSVs, the dataset, or model weights larger than a
few MB. Pin versions in `requirements.txt`.

## 4. Sensor simulator

The simulator is the only source of non-vision data.

Model the conveyor as a fixed structure of configurable length (default 5 km) with a
configurable number of sensor stations (default 10) at fixed kilometre positions. Each
station reports: sensor id, station km, timestamp, temperature, vibration, motor/belt speed,
belt load, motor current.

Behaviour:

- Normal operation with realistic noise, load-dependent correlation between signals
  (more load -> more current and heat), and a slow daily ambient temperature cycle.
- **Gradual degradation, not only sudden spikes.** Prediction is impossible if faults appear
  instantly. Implement named failure modes that develop over time, for example:
  - idler/bearing wear: vibration rises slowly, then temperature follows
  - belt slip / drive problem: speed fluctuation grows, current rises
  - overload: load and current high for sustained periods, temperature climbs
  - sudden events (blockage, spike) as a separate, unpredictable category
- Every reading carries hidden ground-truth columns in the CSV only (fault mode, degradation
  stage, time until simulated failure). These never go to the backend. They exist so the
  model can be evaluated honestly.
- **Messy data, on purpose and configurable:** missing fields, NaN, out-of-range values,
  wrong types, duplicate readings, out-of-order timestamps, a stuck sensor repeating one
  value, and a station that goes silent for a while.
- CLI: `--sensors`, `--interval` (default 5 s), `--speedup` (time acceleration, so hours of
  degradation can be shown in minutes), `--fault-rate`, `--seed`, `--duration`, and a
  `--backfill HOURS` mode that generates history quickly for training and demos.
  Ctrl+C stops it cleanly.
- Writes to CSV and POSTs to the backend. If the backend is down it logs, retries with
  backoff, and keeps running.

## 5. Backend and prediction

### Backend

Endpoints (Swagger docs on by default):

- `POST /api/sensors/readings` (single or batch)
- `GET /api/sensors`, `GET /api/sensors/{id}`, `GET /api/sensors/{id}/history` (paginated)
- `GET /api/anomalies` (paginated, filter by sensor/severity/date), `GET /api/anomalies/active`
- `GET /api/predictions`, `GET /api/predictions/{sensor_id}`
- `POST /api/vision/detect`, `GET /api/vision/detections`, `GET /api/vision/damages`
- `GET /api/data-quality`, `GET /api/dashboard/summary`, `GET /health`

Rules:

- Route handlers stay thin; database work lives in `crud.py`; analysis lives in `ml/`.
- Analysis runs when readings are ingested (synchronously is fine at this scale; say so in
  the design doc and say what you would do at real scale).
- **Bad readings are never silently dropped.** Reject with a clear 422 where the payload is
  unusable; where it is partly usable, store it flagged. Keep a rejected/flagged log with the
  reason, and expose counts per sensor through `/api/data-quality`.
- De-duplicate on (sensor id, timestamp). Accept out-of-order readings and place them correctly.
- A station that has not reported within a timeout is `offline`, which is different from normal.
- An anomaly is "active" until the sensor has been back to normal for a configurable period.

### Anomaly detection

- Isolation Forest on scaled features, trained on simulator backfill data. Include rolling
  features (mean, standard deviation, slope over a recent window) as well as raw values,
  so slow drift is visible to the model.
- Impute missing values in one documented way; do not let a NaN crash inference.
- Each anomaly gets a **type** (which signals drove it), a **severity** (Low / Medium / High /
  Critical, from the anomaly score plus how far readings are beyond operating limits), a
  location and a timestamp.

### Failure prediction: what, where, when

The dashboard must show what might fail, where, and when. Isolation Forest alone cannot do
that, so build a separate, transparent predictor in `ml/risk.py`:

- **What:** map the pattern of drifting signals to a likely failure mode (bearing/idler,
  drive/slip, overload).
- **Where:** the station and its kilometre position.
- **When:** fit the recent trend of each drifting signal and extrapolate to its failure
  threshold. Report an estimated time window (for example "6 to 14 hours"), not a single
  timestamp, and a confidence level based on how well the trend fits and how much data there is.
  When there is no clear trend, say "no estimate" instead of inventing one.
- **Risk level:** Low / Medium / High / Critical, combining the time estimate, recent anomaly
  frequency and severity. Store a plain-language reason with every prediction.
- Station status for the dashboard: `normal`, `abnormal` (active anomaly now),
  `predicted failure` (risk High or Critical), or `offline`.

### Evaluation (real numbers only)

Use the simulator's hidden ground truth:

- Anomaly detector: precision, recall and false-alarm rate against labelled fault periods.
- Predictor: how early the warning came before the simulated failure, and how far the
  estimated window was from the true time.

`python -m ml.evaluation` prints these and writes them to `docs/evaluation_results.md`.
Report whatever the run produces, including bad results. Never write a metric you did not
compute. State plainly that these numbers describe performance on simulated data only.

## 6. Vision component

### Dataset

- Search for public conveyor-belt damage datasets (Kaggle, Roboflow Universe, Mendeley Data,
  GitHub). Actually open each candidate and confirm it exists, what the classes are, whether
  it has bounding-box annotations, how many images, and its licence.
- Show me a short comparison of the candidates and your recommendation, then
  **wait for my choice**.
- Write `scripts/download_dataset.py`. If the download needs an account or API key (Kaggle
  and Roboflow both do), stop and tell me exactly what to set up. Do not work around it and
  do not commit credentials.
- If no dataset with usable annotations can be obtained, tell me. The fallback is a generic
  surface-defect dataset or classical OpenCV detection, clearly labelled as a prototype.
  Do not present a general-purpose pretrained YOLO as a belt damage detector.

### Detection

- OpenCV preprocessing suited to dusty, low-contrast belt images (for example CLAHE and
  denoising), with a short note on why each step is there.
- Fine-tune the YOLO nano model on CPU with a small, stated budget (small image size, few
  epochs, a subset if needed). Record the real validation metrics from the run.
- `POST /api/vision/detect` accepts an image upload and returns boxes, class and confidence.
  Reject non-images and corrupt files cleanly. Save annotated images for the dashboard.
- Model loading is lazy, and a missing weights file produces a clear error, not a crash.

### The continuous-loop problem

The brief says only: "The belt is a continuous loop, and your vision component has to
account for that." Working out the consequences is part of the assessment. Before writing
code, write your analysis in `docs/DESIGN.md` and show it to me. Cover at least:

- **Two reference frames.** Sensor stations are fixed to the structure. Damage is on the
  belt, which moves. A tear has no fixed kilometre along the conveyor; it has a position
  along the belt loop (roughly twice the conveyor length). Do not attach damage to a
  sensor station.
- **Repeat sightings.** A fixed camera sees the same damage once per revolution, so naive
  counting inflates the damage count forever.
- **Useful side of repetition.** Each lap is a fresh measurement of the same damage, so
  growth can be tracked, and damage that stops appearing may have been repaired or missed.
- **Position estimation.** Belt position = distance travelled modulo loop length, from belt
  speed and time. This drifts with slip and speed error. The splice (belt joint) appears
  once per lap and can serve as a zero marker to correct drift; it can also be mistaken for
  damage.
- **Same damage or new damage?** Appearance changes between laps (dust, lighting, material
  on the belt, growth), and two different defects can sit close together.

Then build a damage registry in `vision/belt_registry.py`:

- Each detection gets a belt-loop position and lap number.
- It is matched to an existing registry entry by position within a tolerance, same class,
  and an appearance check (for example perceptual hash or ORB feature match on the crop).
- A match updates the entry (last seen, lap count, size history). No match creates a new one.
- Entries not seen for N laps are marked "not re-observed", not deleted.
- The dashboard shows unique damages, not raw detections, with a growth trend per damage.

Because the dataset is unrelated still images and not footage of one belt, simulate the
loop: `scripts/seed_demo.py` places a set of dataset images at belt positions and replays
them over several laps with small changes (brightness, shift, blur, speed jitter), plus
clean frames. This must be stated openly in the README. State what stays unsolved without
a real encoder, real camera calibration and real multi-lap footage.

## 7. Dashboard

Streamlit + Plotly, all data from the backend API, nothing hardcoded. Required views:

1. **Belt status:** every sensor station on a line by kilometre, coloured normal / abnormal /
   predicted failure / offline, with latest readings.
2. **Active anomalies:** location, type, severity, time.
3. **Predictions:** what might fail, where, estimated time window, confidence, reason.
4. **Vision:** unique damages with annotated image, class, confidence, belt-loop position,
   times seen, growth trend.

Also: trend charts per sensor, a small data-quality panel (rejected/flagged counts), filters
for sensor, severity and date, manual and auto refresh, and clear messages when the backend
is down or there is no data yet. Keep it readable; do not spend time on decoration.

## 8. Tests

Pytest, no GPU, no network, no large model downloads. Cover: simulator output and its messy
modes, each kind of bad input at the API, de-duplication and out-of-order handling,
preprocessing with missing values, anomaly detection on a known-bad reading, time-to-failure
on a synthetic ramp with a known answer, image preprocessing, and the damage registry
(same damage across laps counted once; two nearby damages kept separate; position drift
within tolerance). Mock the YOLO model in tests.

## 9. Documentation

Write these in plain, direct language. No marketing words. I have to defend every sentence
in an interview.

**`docs/DESIGN.md`** (linked from the README): what is monitored and why those signals, the
failure modes considered, architecture (Mermaid diagram), data flow, how prediction works,
the loop analysis, and assumptions made where the brief was unclear.

**`README.md`**, in this order:
1. What this is, in three or four sentences
2. Quick start for Windows (exact commands)
3. Approach, explained
4. Design decisions, each with the alternative considered and why it lost
5. Vision dataset: source, licence, why chosen, what was rejected
6. The continuous-loop problem: consequences, what is handled, what is not
7. How messy data is handled
8. Built / simplified / skipped table, with a reason per row
9. Evaluation results (real, from `docs/evaluation_results.md`) and their limits
10. Dashboard screenshots (leave placeholders; I will take them)
11. What I would do differently
12. How AI tools were used (points to `ai_usage/`)

**`ai_usage/`**: create `README.md` explaining what is in the folder, and `links.md` as an
empty template. I will paste the real chat history myself. Do not invent links or logs.

**`docs/DECISIONS.md`**: keep a running log as you work. One entry per significant decision:
what, options, choice, reason. Also log every suggestion of yours that I rejected or changed.

## 10. Commands (must work from the project root on Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m simulator.sensor_simulator --backfill 48
python -m ml.train
python -m ml.evaluation
uvicorn backend.main:app --reload
python -m simulator.sensor_simulator --speedup 60
python scripts/seed_demo.py
streamlit run dashboard/app.py
pytest -v
```

`run.py` starts backend, simulator and dashboard together for a demo.

## 11. How to work

Phases: (1) structure and config, (2) simulator, (3) backend, (4) anomaly detection and
prediction, (5) vision, (6) dashboard, (7) integration run, (8) tests, (9) docs.

For each phase:

- Start by telling me in a few lines what you plan to build and any decision you want my
  input on. For phases 4 and 5, wait for my answer before coding.
- Build it, run it, and fix what breaks. Show me the real output of the run.
- Commit to git with a clear message at the end of the phase.
- Report: files created or changed, what was verified and how, what is not working or not
  done, and how I can test it myself.

Rules:

- Create real files; no placeholders or TODOs in core paths.
- Do not hardcode results, fake predictions, or invent metrics, dataset facts or links.
- Do not say something works unless you ran it. If you could not run it, say that.
- Check existing files before overwriting anything.
- When the brief is unclear, make a reasonable assumption and record it in `docs/DESIGN.md`.
- Prefer the simple, reliable option. Ask me before any major architectural change.

Begin with phase 1: inspect the workspace, save this spec, create the structure, then tell
me your plan for phase 2.
