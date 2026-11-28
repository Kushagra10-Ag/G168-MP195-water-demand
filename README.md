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
## How to run

Install dependencies:

    pip install -r requirements.txt

Run tests:

    pytest

Start the API:

    uvicorn app.main:app --reload

API documentation: http://127.0.0.1:8000/docs

Docker image:

    docker build -t mp195-api:1.0 .

Docker run:

    docker run -p 8000:8000 mp195-api:1.0

Kubernetes:

    kubectl apply -f k8s/

Endpoints:
- GET /health
- POST /predict

Validation rules:
- temp_c <= 75
- garden_area_m2 >= 0