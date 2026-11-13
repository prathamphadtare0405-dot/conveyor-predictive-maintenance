# Decision log

One entry per significant decision: what was decided, the options, the choice and the
reason. Entries marked **[author]** record a suggestion from the AI assistant that the
author rejected or changed.

Scope cuts are logged here too, with the reason.

---

## Phase 1: structure and config

### D1. Project location
- **Options:** build inside the user home folder directly; a new folder under it.
- **Choice:** a new folder `conveyor-predictive-maintenance` under the home folder.
- **Reason:** the working directory was the home folder, which is not a project. A separate
  folder keeps the repo self-contained and outside OneDrive (file sync can lock a SQLite
  file while it is being written).

### D2. Python version
- **Options:** install Python 3.11 or 3.12; use the Python 3.13.9 already on the machine.
- **Choice:** 3.13.9 in a `.venv`.
- **Reason:** the spec asks for 3.11+, and every package in the stack installed and passed
  `scripts/check_env.py` on 3.13. The project has only been run on 3.13.9; 3.11 and 3.12
  are untested.

### D3. How configuration is loaded
- **Options:** `pydantic-settings`; a plain frozen dataclass reading `os.getenv` with
  `python-dotenv` for the `.env` file.
- **Choice:** dataclass + `python-dotenv`, in `backend/config.py`.
- **Reason:** the simulator, ML code and dashboard all need the config, and they should not
  need FastAPI/pydantic machinery to read ten numbers. Bad values still fail at start-up
  with the variable name in the error.

### D4. One shared table of signal limits
- **Options:** each component defines its own thresholds; one table in `backend/config.py`.
- **Choice:** one `SIGNALS` table (physical range, warning limit, failure limit, direction).
- **Reason:** the simulator decides when a component has "failed", the backend validates
  ranges and grades severity, and the predictor extrapolates to the failure limit. If these
  used different numbers the evaluation would be meaningless.
- **Consequence to be honest about:** the predictor knows the same failure threshold the
  simulator uses. On a real belt the threshold would come from the equipment maker or from
  failure history and would be less exact. This is stated in the evaluation limits.

### D5. Signal limits are constants, other settings are environment variables
- **Reason:** limits are engineering data that belong in version control with their
  reasoning (see `docs/DESIGN.md`); deployment settings (ports, paths, timeouts, rates)
  change per machine and belong in the environment.

### D6. Extra packages beyond the listed stack
- `python-dotenv` (load `.env`), `python-multipart` (FastAPI needs it for image upload),
  `httpx` (FastAPI's test client needs it), `requests` (simulator and dashboard HTTP calls),
  `uvicorn` (to serve FastAPI). All are small supporting libraries, not stack changes.

### D8. Some packages pinned below the latest release (Windows Smart App Control)
- **What happened:** with the newest versions installed, `import torch`, parts of
  scikit-learn, SQLAlchemy, pyarrow, polars and fonttools failed with "An Application
  Control policy has blocked this file". Smart App Control is on in enforce mode on the
  development laptop and blocks compiled files it has no reputation for, which hits very
  recent wheels.
- **Options:** turn Smart App Control off (it cannot be turned back on without reinstalling
  Windows); use releases that the policy already allows.
- **Choice:** keep the security setting; pin torch 2.8.0, torchvision 0.23.0,
  scikit-learn 1.7.2, SQLAlchemy 2.0.43, pyarrow 21.0.0, polars 1.34.0, fonttools 4.60.1.
- **How it was checked:** every `.pyd`/`.dll` in the venv was test-loaded to find the
  blocked ones, then `scripts/check_env.py` exercised each library (7/7 checks pass).
- **Cost:** the pins are older than necessary on a machine without this policy. Nothing in
  the project needs a feature newer than these versions.

### D9. Git installed during setup
- Git was not on the machine. Installed Git for Windows with `winget` so that each phase
  can be committed, as the spec requires.

### D7. Empty module files are not created up front
- **Options:** create every file in the spec's tree now as stubs; create each file in the
  phase that implements it.
- **Choice:** create each file when it has real content.
- **Reason:** the spec forbids placeholders in core paths, and a commit history where files
  appear with their implementation is easier to review.

---

## Phase 2: simulator

### D10. Faults adjust healthy values; they do not generate their own
- **Options:** separate signal generators per fault; one healthy model whose output a fault modifies.
- **Choice:** one healthy model; a fault first changes the load, then the other signals.
- **Reason:** a fault on a real belt sits on top of normal load swings and the day/night
  cycle. Keeping those underneath the fault is what makes detection non-trivial.

### D11. Fault growth is `progress ** shape`, with a random shape between 1.0 and 2.2
- **Options:** linear growth only; exponential growth; a random power curve.
- **Choice:** random power curve per fault.
- **Reason:** the predictor extrapolates a linear trend. If the simulator only produced
  linear drift, the evaluation would be a self-fulfilling test. Curved faults make the
  linear fit over-estimate the time left early on, and the evaluation shows by how much.

### D12. The simulated failure time is fixed when the fault starts
- **Reason:** it gives every reading an exact "time until failure" label. The failing
  signal is driven to its failure limit at that moment (vibration for bearing wear, speed
  for slip, temperature for overload).
- **Limit:** real components do not fail at an exact threshold. Stated in the evaluation.

### D13. Load is fed at the tail and reaches each station after its travel time
- **Options:** independent random load per station; one feed profile with transport delay.
- **Choice:** shared feed profile, delayed by km / belt speed, plus small per-station noise.
- **Reason:** it is the same material passing each station, and it costs one line of code.

### D14. Messy data is applied after the true values are computed, and is labelled
- Each damaged reading carries `gt_messy` in the CSV (never sent to the backend), so tests
  and evaluation can tell a corrupted reading from a fault.
- A silent station writes nothing to the CSV either: the CSV is a record of what was emitted.
- "Missing field" removes the timestamp in about 15 % of cases, so the backend's reject
  path gets exercised as well as its flag path.

### D15. `--speedup` shortens the wait between ticks; it does not change the sampling interval
- **Options:** keep the tick rate and make each tick cover more simulated time; keep 5 s
  samples and send them faster.
- **Choice:** keep 5 s samples.
- **Reason:** the rolling features and trend fits are tuned to the sampling interval. If
  speed-up changed it, the live demo would not behave like the data the model was trained on.
- **Consequence:** in live mode simulated time runs ahead of the wall clock. The backend
  therefore measures "offline" by arrival time, and everything else by reading timestamps.

### D16. Backfill does not post to the backend unless `--post` is given
- **Reason:** 48 hours is about 340 000 readings. Its purpose is training data. Sending it
  through synchronous analysis would take a long time for no benefit.

### D17. Additions to the CLI beyond the spec
- `--inject MODE:SENSOR:HOURS` starts a chosen fault at once (so a demo does not wait for a
  random one), `--messy-rate`, `--messy-kinds`, `--csv`, `--backend-url`, `--post`, `--no-post`.
- `--duration` is in real seconds (how long the process runs), not simulated seconds.

### D18. Backfill ends at the current time, so two runs with the same seed differ in timestamps
- The fault sequence is the same for a given seed, but the day/night phase depends on when
  the run happens. `ml.evaluation` passes a fixed end time so its numbers are repeatable.

---

## Phase 3: backend

### D19. Incoming readings are validated item by item, not by a strict request model
- **Options:** a Pydantic model for the request body; a loose body checked by hand per item.
- **Choice:** loose body, checked in `backend/validation.py`.
- **Reason:** with a strict model, one NaN or one text value makes FastAPI refuse the whole
  request, including the good readings in the batch. The requirement is to keep what is
  usable and to record what was not.

### D20. What is rejected and what is flagged
- **Rejected (not stored):** not an object, no sensor id, no or unparseable timestamp, no
  usable signal at all, or a first reading from an unknown sensor without a position.
- **Flagged (stored, bad signal set to NULL):** missing field, NaN/infinity, text or other
  wrong type, value outside the instrument's possible range, station_km disagreeing with
  the registered position, a stuck signal.
- **Reason for the line:** a reading needs a sensor and a time to be placed in the history.
  Everything else can be missing and the rest is still information.
- A number sent as text ("45.2") is used and still logged. Text with a unit ("45.2C") is
  not parsed: guessing units is how silent errors start.

### D21. Bad values become NULL at ingest; filling gaps is left to the analysis code
- **Options:** impute at ingest and store the filled value; store NULL.
- **Choice:** store NULL.
- **Reason:** the database should say what was measured. An imputed value stored next to
  real ones cannot be told apart later.

### D22. HTTP status for batches
- 200 when at least one item was stored or recognised as a duplicate; the body lists every
  problem by item index. 422 when nothing in the request was usable (a single bad reading,
  or a batch of only bad ones), with the same body.
- **Alternative considered:** 207 Multi-Status. Rejected because few clients handle it and
  the body already carries the per-item outcome.

### D23. Duplicates: first reading wins
- A second reading for the same (sensor, timestamp) is not stored and is logged, even if its
  values differ. The alternative, last-write-wins, would let a late duplicate change history
  that analysis has already used.

### D24. Two clocks: data time and arrival time
- History order, duplicates and analysis use the reading's own timestamp. "Offline" uses the
  backend's clock at arrival. A late reading is stored at its own timestamp and marked
  `arrived_late`; the station's "latest reading" stays the one with the newest timestamp.
- **Not done:** readings with timestamps far in the future are not rejected, because the
  simulator's time acceleration produces exactly that. A real deployment would reject them.

### D25. Stuck-sensor rule
- A signal that repeats exactly the same value 6 times in a row (configurable) is flagged
  from the 6th reading on. The first five repeats are not flagged after the fact.
- The value is kept in the database with the flag; the analysis code treats it as missing.
- Late readings are skipped by this check, since "the readings just before" is ambiguous for them.

### D26. `backend/validation.py` is an addition to the spec's file list
- Validation is neither routing nor database work, and the training code needs the same
  cleaning rules for CSV data, so it has its own module with no database dependency.

### D27. Anomaly, prediction and vision tables and routes are added in their own phases
- Phase 3 covers ingest, storage, sensors, data quality, summary and health. Station status
  is `normal` or `offline` for now; `abnormal` and `predicted failure` arrive with phase 4.

---

## Phase 4: anomaly detection and prediction

### D28. The model is trained on healthy readings only
- **Options:** (A) train on rows the simulator labels healthy; (B) train on everything, unlabelled.
- **Choice:** A. The assistant recommended A; the author asked which was simpler and accepted it.
- **Reason:** about 20 % of the backfill is fault data, far too much for an Isolation Forest
  to treat as rare. A corresponds to training on a commissioning period signed off as
  fault-free. The labels are used only to pick training rows; evaluation uses a separate run.

### D29. Rolling window of 360 readings (30 minutes), not a couple of minutes
- **Reason:** faults develop over hours. Over 2 minutes the slope of a noisy signal is
  noise; over 30 minutes a drift of a few tenths of a unit per hour is visible.

### D30. A baseline removes what load and time of day explain
- **What:** a least-squares fit on healthy data predicts temperature, vibration, speed and
  current from load (raw and 10-minute mean) and hour of day; each station has its own
  offset. Features and trend fits use the residual (measured minus expected).
- **Alternative:** work on raw signals. Rejected: temperature rises several degrees every
  morning and with every load swing, and a trend fit on the raw signal reads that as a fault.
- **Limit:** hour of day stands in for ambient temperature. A real installation should
  measure ambient temperature.

### D31. Three rules decide whether a reading is anomalous, and the forest is the weakest
- **Found during testing:** the Isolation Forest did not flag a bearing whose vibration had
  doubled but was still below the warning limit. A tree can only split inside the range of
  its training data, so a value far outside it scores the same as the most extreme healthy
  value.
- **Options:** loosen the forest's threshold (more false alarms everywhere); add a rule
  for exactly this case.
- **Choice:** keep the forest as specified and add a drift rule: the rolling mean of the
  load-compensated residual is outside the healthy band by 4 healthy standard deviations.
  Warning-limit violations are the third rule.
- **Result (see `docs/evaluation_results.md`):** recall on fault readings went from 0.71 to
  0.92 with the same false-alarm rate. The forest alone, where neither other rule fired,
  flagged 0.2 % of fault readings. The report states this; it is not hidden.

### D32. An anomaly opens after 3 anomalous readings in a row, or at once if severity is High or Critical
- **Reason:** one odd reading should not open an anomaly, but a limit violation should not wait.
- **Cost:** a single-reading spike that stays inside limits is deliberately ignored.

### D33. Severity
- Low: forest only, just below threshold. Medium: forest well below threshold, or drift.
  High: forest far below threshold, or a signal beyond its warning limit. Critical: a signal
  at or beyond its failure limit.

### D34. Prediction is rules and straight-line fits, not a learned model
- **Options:** train a regressor on the simulator's time-to-failure labels; extrapolate trends.
- **Choice:** trend extrapolation in `ml/risk.py`.
- **Reason:** a regressor trained on simulator labels would learn the simulator. Every
  number the trend method reports can be recomputed by hand from the chart, and the reason
  text says which numbers were used.
- **Known weakness:** real wear accelerates, and a straight line then over-estimates the
  time left. The early end of the window uses the slope of the most recent third of the
  history to compensate, which is only partly enough: in the evaluation the failure came
  sooner than the window in about a quarter of estimates, mostly when more than 4 hours remained.

### D35. Risk level is the highest of three readings of the situation
- Time (early end of the window: 2 h Critical, 8 h High, 24 h Medium), level (how far the
  primary signal has moved from normal to the failure limit), and the anomaly record of the
  last hour. A low-confidence time estimate counts one level lower.
- **Consequence:** risk stays raised for up to an hour after a repair, because the anomaly
  record still counts.

### D36. Evaluation data is the same conveyor on different days
- **Found during evaluation:** with a different simulator seed the false-alarm rate was 16
  times the design value. The seed also changed each station's permanent sensor offsets, so
  the test data came from a different conveyor than the training data.
- **Fix:** a separate `plant_seed` fixes the station offsets; `seed` changes load, noise,
  faults and messy data. Training, evaluation and demo share the plant seed.
- The first evaluation run also contained no belt-slip fault. One fault of each mode is now
  injected at the start of the evaluation run, on top of the random ones.

### D37. The thresholds in `ml/risk.py` were set before looking at evaluation results and not tuned afterwards
- The drift rule (D31) and the plant seed (D36) were changed after seeing evaluation output,
  because they fixed defects. The numeric thresholds of the predictor were left alone, so the
  window accuracy in the report is not the result of fitting to the test run.

### D38. Live analysis speed
- Analysis runs inside the ingest request and takes about 160 ms per request on the
  development laptop, plus about 40 ms per prediction. The simulator groups readings so that
  it sends at most two requests per second. With `--speedup 60` the measured rate was about
  36 times real time; the simulator slows down and loses nothing.
- **Not done:** an in-memory cache of each station's recent readings would remove most of
  the cost. Left out because real-time operation (one batch every 5 seconds) is nowhere
  near the limit.

### D39. Late readings older than 10 minutes are stored but not scored
- Scoring them would mean rebuilding old rolling windows for an alarm about the past.

### D40. `ml/live.py` is an addition to the spec's file list
- `ml/predict.py` holds the detector with no database access; `ml/live.py` connects
  detector and predictor to the backend's tables.

---

## Phases 6 and 7 (sensor side): dashboard and integration

Built before the vision phase because the vision dataset needs a decision and credentials
from the author. The vision view is added with phase 5.

### D41. Status is shown by colour, shape and word together
- Normal is a green circle, abnormal a yellow diamond, predicted failure a red triangle,
  offline a grey cross, each with its label. Colour alone fails for colour-blind readers
  and on a projector.

### D42. Trend charts are one chart per signal
- **Alternative:** all five signals on one chart with two y-axes. Rejected: the signals have
  different units and scales, and a second axis invites reading crossings that mean nothing.
- Warning and failure limits are drawn only when they are near the data, so a limit far
  away does not flatten the line.

### D43. The date filter is a period ending at the latest reading, not a calendar picker
- With time acceleration the readings' timestamps run ahead of the wall clock. "Last 3
  hours" of data is unambiguous; a calendar date is not.

### D44. Auto refresh re-runs the whole page
- **Alternative:** partial refresh of each panel. Rejected as unnecessary: one full refresh
  is six API calls.

### D45. Checked without a browser
- The dashboard was run headlessly with Streamlit's test runner in three states (backend
  down, no data, live data with two faults) and raised no exceptions; `run.py` was run and
  all three services answered. It has not been looked at in a browser by the assistant;
  the author takes the screenshots.

### D46. `run.py` binds the dashboard to localhost
- Streamlit listens on all network interfaces by default. A demo does not need that.

---

## Phase 5: vision

### D47. Dataset: crack-seg (cracks in roads and walls), used as a stand-in
- **Candidates looked at:**
  - Roboflow Universe projects "conveyor belt tear", "CBDD" and others. Their pages
    returned HTTP 403 to automated access, so classes, sizes and licences could not be
    confirmed, and downloading needs an account and API key.
  - BeltCrack (GitHub UESTC-nnLab/BeltCrack, paper arXiv 2506.17892): real belt images in
    sequences, 14,087 and 9,645 images. The repository page loads and is Apache-2.0, but
    the data is on ScienceDB, whose page could not be read automatically; annotation
    format, size and the dataset's own licence stayed unconfirmed.
  - Mendeley "In-lab image dataset of foreign objects and anomalies in iron ore conveyor
    belts" (CC BY 4.0): foreign objects on a lab belt, not belt damage.
  - Kaggle: no belt-damage dataset found; needs an account and API key anyway.
  - crack-seg, published by Ultralytics: 4,029 photographs of cracks in roads and walls,
    one class, segmentation masks, marked Public Domain on its documentation page, direct
    download with no account.
- **Author's instruction:** everything must be free; first "no API keys", then "free API
  keys are fine". The assistant had recommended the Roboflow belt-tear set (needs a key).
- **Choice:** crack-seg. It is the only candidate whose facts could be verified and which
  downloads without anyone creating an account. The counts were checked after download
  (3,717 / 200 / 112 images, one class).
- **What this costs:** the detector is a crack detector for masonry and asphalt. It is not
  a belt damage detector and is not presented as one. This is the spec's stated fallback
  ("a generic surface-defect dataset ... clearly labelled as a prototype").
- **How to switch:** put a belt dataset in YOLO format under `data/datasets/`, point
  `vision/dataset.py` at it and retrain. Nothing else depends on the dataset.

### D48. Segmentation polygons are converted to bounding boxes
- The API has to return boxes, and box detection is cheaper to train on CPU than
  segmentation. A box around a thin diagonal crack is mostly background; this is stated in
  the training report.

### D49. Training budget
- YOLO11n, 800 of 3,717 training images (seeded subset), the whole validation split,
  image size 320, 12 epochs, CPU. Chosen to finish in roughly ten minutes on the laptop.

### D50. The same preprocessing is applied to training images and at inference
- Bilateral denoise, then CLAHE on the lightness channel (reasons in
  `vision/preprocessing.py`). Training on unprocessed images and then feeding processed
  ones would show the model images unlike its training data.

### D51. Appearance check: difference hash, 256 bits
- **Options:** perceptual (difference) hash of the crop; ORB keypoint matching.
- **Choice:** difference hash, match when at least 65 % of bits agree.
- **Reason:** it is a few lines, has no tunable detector, and tolerates brightness change,
  blur and small shifts (tested). ORB finds few keypoints on dark, low-texture rubber.
- **Limit:** it compares coarse structure only. Two similar-looking cracks close together
  can be merged, and a damage that changes a lot between laps can be split in two.

### D52. One-to-one matching applies within a frame, not within a lap
- Two boxes in one image are two different things, so an entry takes at most one of them.
- Two frames of the same pass may show the same damage (overlapping frames), so a second
  match in the same lap is allowed; it adds a sighting but not a lap.
- **Changed from the analysis text**, which first said "one to one per lap".

### D53. Position tolerance grows with distance from the reference marker
- tolerance = 5 m + 0.0005 x metres travelled since the marker (10 m at the far end of the
  loop). Position error accumulates since the last correction, so a fixed tolerance is
  either too loose near the marker or too tight far from it.

### D54. "Not re-observed" counts passes, not laps
- A damage is only counted as missed in the current lap once the camera has gone past its
  position by more than the tolerance. Otherwise everything ahead of the camera would be
  marked missing at the start of each lap.

### D55. The client sends the belt position; the backend only falls back to dead reckoning
- The position tracker belongs with the camera and the speed sensor. The backend accepts
  `belt_position_m` and `lap` with each frame. Without them it estimates position from the
  capture time at nominal speed and labels the result `dead_reckoning`.

### D56. Frames without detections are not stored as images
- Only the frame row is kept (so "passes missed" can be counted). Annotated images are
  saved only for frames that show damage.

### D57. `POST /api/vision/reset`, `vision/train.py` and `scripts/check_env.py` are additions to the spec
- Reset empties the damage registry so the demo can be replayed; it would need access
  control in a real deployment. `vision/train.py` holds the training run and writes
  `docs/vision_training_results.md`.

### D58. The loop is simulated, with a calibration lap
- `scripts/seed_demo.py` replays test-split images at fixed belt positions over several
  laps (details in its docstring). The simulated speed sensor reads 0.4 % high; the first
  lap sends nothing and only measures that error between two marker events. Without that
  lap, the first lap's positions would be up to 40 m off and every damage would be
  registered twice.

### D59. The appearance check was replaced after the first loop replay (supersedes D51)
- **What happened:** with the difference hash of the box crop, the replay turned 7 placed
  damages into 19 registry entries. Position tracking was fine (largest error 3.5 m). The
  cause was the detector: it does not draw the same box around the same crack twice, and a
  hash of the crop changes when the crop does.
- **Measured** (40 test images, 5 varied "laps" each, real detector): at its threshold of
  0.65 the hash accepted 46 % of true repeat sightings. A template comparison (is the stored
  box content found in the new detection's surroundings, or the reverse; normalised
  cross-correlation over six scales) accepted 97 % at a threshold of 0.55 and wrongly
  accepted 0.4 % of pairs of different cracks.
- **Choice:** template comparison in `vision/appearance.py`, threshold 0.55. The hash stays
  as the registry's default comparison for unit tests, because it needs no images.
- **Result of the same replay:** 45 raw detections became 10 registry entries for 7 placed
  images. In two images the detector found two cracks in the same frame, which the
  registry correctly keeps apart (9 distinct things). One damage was still registered
  twice (seen as one entry on laps 1, 2, 6, 7 and as another on laps 3, 4, 5). The repaired
  damage ended as "not re-observed" and the late one was first registered on lap 3.
- **The threshold was chosen on the same kind of replayed images it is then demonstrated
  on.** It is a starting value, not a validated one.

### D60. A damage is "confirmed" after two laps
- One sighting may be dirt, a shadow or a detector error. The dashboard counts confirmed
  and unconfirmed damages separately. The cost is one lap (42 minutes) of delay.

### D61. Growth below 5 % per lap is reported as stable
- In the replay, damages that did not change showed trends of up to about 3 % per lap from
  box jitter alone. One damage scripted to grow 7 % in scale per lap was reported as
  growing at +19 % of its first box area per lap; another box on the same image was
  reported as shrinking. Box area from this detector is a noisy size measure.
