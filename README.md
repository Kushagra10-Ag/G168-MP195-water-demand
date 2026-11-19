# G168 - MP195 Housing Society Water Demand (Kochi)

- Group ID: G168
- Problem ID: MP195
- Faculty: Richa Sharma, Batch 26

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU0056 | Kushagra Agrawal |
| S24CSEU0157 | Vaibhav Sunola |
| S24CSEU0129 | Ayush Sharma |
| S24CSEU0079 | Tejaswita Tiwari |

## What this project does

Predicts daily water use (litres_per_day) of a housing society in Kochi,
with data cleaning, MLflow tracking, a FastAPI service, Docker, Kubernetes files,
CI, drift monitoring and retraining.

## Setup

Use Python 3.12 and run commands from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.
The pinned `requirements.txt` is supplied by Ayush.

Copy the original `train.csv` and `later.csv` into `data/raw/`.
If a DVC remote is configured, install DVC (`pip install dvc`) and use
`dvc pull` instead. Do not edit or save the raw CSVs in Excel.

## Vaibhav's Stage 1 checks

```powershell
python src/mp195/data_checks_b.py
python -m pytest tests/test_checks_b.py
```

Tests use small in-memory datasets and need Tejaswita's `pytest.ini`.
Until that file is available, run `python -m pytest -o pythonpath=src tests/test_checks_b.py`.
The script reports suspected leaking columns and target typos without changing the data.
