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
