"""Check that every compiled dependency loads and works on this machine.

Run this after `pip install -r requirements.txt`:

    python scripts/check_env.py

It exists because Windows Smart App Control can block individual compiled files inside
freshly released wheels ("An Application Control policy has blocked this file"). Each
check exercises one library for real, so a blocked file shows up here and not in the
middle of a training run.
"""

from __future__ import annotations

import sys
import traceback


def check_numpy_pandas() -> str:
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"a": np.arange(10.0)})
    return f"numpy {np.__version__}, pandas {pd.__version__}, rolling mean {df['a'].rolling(3).mean().iloc[-1]}"


def check_sklearn() -> str:
    import numpy as np
    import sklearn
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    x = np.random.default_rng(0).random((300, 4))
    labels = IsolationForest(random_state=0).fit(StandardScaler().fit_transform(x)).predict(x[:3])
    return f"scikit-learn {sklearn.__version__}, IsolationForest predict {labels.tolist()}"


def check_sqlalchemy() -> str:
    import sqlalchemy as sa

    with sa.create_engine("sqlite://").connect() as conn:
        value = conn.execute(sa.text("select 1")).scalar()
    return f"SQLAlchemy {sa.__version__}, select 1 -> {value}"


def check_fastapi() -> str:
    import fastapi
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return f"fastapi {fastapi.__version__}, GET /health -> {TestClient(app).get('/health').json()}"


def check_opencv() -> str:
    import cv2
    import numpy as np

    out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.zeros((64, 64), np.uint8))
    return f"opencv {cv2.__version__}, CLAHE output {out.shape}"


def check_torch_yolo() -> str:
    import torch
    import torchvision
    import ultralytics

    total = float(torch.ones(3).sum())
    return (
        f"torch {torch.__version__} (cuda={torch.cuda.is_available()}, sum={total}), "
        f"torchvision {torchvision.__version__}, ultralytics {ultralytics.__version__}"
    )


def check_dashboard_libs() -> str:
    import pandas as pd
    import plotly
    import pyarrow as pa
    import streamlit

    rows = pa.Table.from_pandas(pd.DataFrame({"a": [1, 2]})).num_rows
    return f"streamlit {streamlit.__version__}, plotly {plotly.__version__}, pyarrow {pa.__version__} ({rows} rows)"


CHECKS = [
    check_numpy_pandas,
    check_sklearn,
    check_sqlalchemy,
    check_fastapi,
    check_opencv,
    check_torch_yolo,
    check_dashboard_libs,
]


def main() -> int:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    failed = 0
    for check in CHECKS:
        name = check.__name__.removeprefix("check_")
        try:
            print(f"[ok]   {name}: {check()}")
        except Exception as exc:  # report every failure, do not stop at the first
            failed += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
