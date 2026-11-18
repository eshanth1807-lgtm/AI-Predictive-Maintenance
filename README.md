# MachineWatch — AI-Based Industrial Equipment Failure Detection

A complete laptop-based prototype using the UCI AI4I 2020 dataset. Includes
model comparison, a trained classifier, a Streamlit dashboard, a FastAPI service,
SQLite prediction history, automated tests, an exploration notebook and a research report.

**Scope:** classify the failure label at a recorded operating condition. This is
not a time-to-failure forecast or a validated industrial safety system. No hardware is required.

## Start the dashboard

Use Python 3.12. From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. The dataset and trained model are included.
On Windows, activate with `.venv\Scripts\activate` instead.
An already-prepared `.venv` is available on the original build machine:
use `.venv/bin/python -m streamlit run app.py` directly.

If Python is unavailable on macOS, install Python 3.12 from https://www.python.org/downloads/.
The app does not train at startup. If model loading reports incompatible package
versions, install the pinned requirements or retrain using your environment.
Only load trusted local joblib artifacts; never load a model uploaded by a stranger.

## Reproduce the experiment

```bash
python scripts/download_data.py
python train.py
python -m pytest -q
```

Training regenerates the model, metrics, plots, split assignments, test predictions,
false-negative table and PDF report. The downloader leaves an existing dataset unchanged.
`requirements.lock.txt` captures the full tested environment; `requirements.txt`
pins direct dependencies. Downloading/installing requires internet access; inference is local.

## How the model is evaluated

- Six inputs only: Type, air temperature (K), process temperature (K), rotational
  speed (rpm), torque (Nm), and tool wear (min).
- UDI, Product ID, Machine failure and all five failure-mode columns are excluded
  from features. Failure modes appear only in post-test analysis.
- Stratified 60% train / 20% validation / 20% test split with seed 42.
- Encoding and scaling are fitted on training records only.
- Prior baseline, balanced logistic regression, decision tree and random forest.
- Each learned model's threshold maximizes validation F2 on a fixed grid; the
  highest validation F2 selects the model. F2 emphasizes recall. No test tuning.
- Final held-out accuracy, precision, recall, F1, F2, average precision, ROC AUC,
  confusion matrix, false-negative analysis and recall bootstrap interval.
- Saved model is exactly the training-fitted model evaluated on test data; no later refit.

See `reports/metrics.json` for actual results and `research_report.pdf` for interpretation.
The included random forest uses threshold **0.21**: test recall **83.82%**, precision
**51.35%**, F1 **0.637**, and accuracy **96.75%**. It detects **57/68** failures,
misses **11**, and produces **54** false alarms. All **20 automated tests** passed after the error-handling review. See `reports/screenshots/dashboard.jpg` for the live preview.
Scores are **uncalibrated model scores**, not real-world failure probabilities.
The threshold is a research choice, not a measured maintenance cost optimum.
Synthetic random-walk measurements can be correlated across random splits; real
deployment requires chronological and machine-held-out evaluation.

## Dashboard features

1. Single-condition form with model score, classification and out-of-range warnings.
2. CSV batch upload, strict six-column template, up to 1,000 rows and downloadable output.
3. Model comparison, confusion matrix, precision-recall curve and global importance.
4. Password-protected admin analytics and private prediction history.

Input limits are broad plausibility bounds. A value inside these limits but outside
the training range receives a warning; it is not evidence of reliable extrapolation.

## FastAPI backend

In a second terminal, activate the environment and run:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Interactive documentation: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"machine_type":"M","air_temperature":300.5,"process_temperature":310.2,"rotational_speed":1500,"torque":42,"tool_wear":120}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model readiness; 503 if unavailable |
| `GET /model` | Model version, threshold, feature names and training ranges |
| `POST /predict` | Predict one condition and save history |
| `POST /predict/batch` | Predict `{"records": [...]}` atomically, 1–1,000 records |
| `GET /history?limit=100` | Recent predictions; maximum 1,000 |

All single-input fields have defaults for interactive examples; send all six for
actual use. Unknown fields, nonfinite values, invalid type and out-of-bounds values
are rejected. Both interfaces share `maintenance/predictor.py` and SQLite.
Set `MAINTENANCE_DB=/path/to/history.sqlite3` to change the database location.

**Local use:** Prediction endpoints are public; history and usage require admin authentication. Bind to localhost. Before exposing it,
add authentication, request limits, TLS and protected persistent storage. Uploaded
CSV records are processed locally; submitted predictions are stored in SQLite.

## Notebook

Open `notebooks/experiments.ipynb` in VS Code with its Jupyter extension and select
the project environment. Install notebook support with
`python -m pip install -r requirements-notebook.txt`. It explores class imbalance, distributions and saved
results; running `python train.py` is the canonical reproducible experiment.

## Project layout

```text
app.py                    Streamlit dashboard
api.py                    FastAPI routes
train.py                  Reproducible training and evaluation
maintenance/              Shared validation, inference and SQLite storage
data/ai4i2020.csv          Original attributed dataset
model/classifier.joblib   Trained preprocessing + classifier bundle
notebooks/experiments.ipynb
reports/                  Metrics, plots, split manifest and error analysis
research_report.pdf       Technical research report
tests/                    Model, validation, API, persistence and UI tests
scripts/                  Dataset download and report generation
```

## Publish the source on GitHub

Create an empty repository in your own GitHub account, then run from this folder:

```bash
git init
git add .
git commit -m "Build predictive maintenance prototype"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Predictive-Maintenance.git
git push -u origin main
```

The `.gitignore` excludes the virtual environment and prediction database. Review
files before pushing. A GitHub Actions workflow retrains and tests on Python 3.12.
No repository or cloud deployment is created automatically by these instructions.

Optional local container: `docker build -t machinewatch .`, then
`docker run --rm -p 127.0.0.1:8501:8501 machinewatch`.
Container history is ephemeral unless you mount a writable persistent data directory.

## Attribution

AI4I 2020 Predictive Maintenance Dataset (2020). UCI Machine Learning Repository.
https://doi.org/10.24432/C5HS5C. Dataset license: CC BY 4.0.
See `data/README.md` for source and license links.

## Private admin dashboard

Start locally with a password prompt (the password is not saved to a file):

```bash
.venv/bin/python scripts/start_dashboard.py
```

Open `http://127.0.0.1:8501/?admin=1` and enter the same password.
Visitors see no admin navigation. The admin menu appears only in an authenticated
session and disappears on logout or expiry. For hosting, append `?admin=1` to your
app URL. This link is a login entry point, not a security secret; the password still
protects access. Anyone who knows the password can authenticate.
There is no default password. Login expires after 30 minutes; sign out explicitly
on shared computers. Changing the configured password invalidates existing sessions.
For hosted use, configure `MAINTENANCE_ADMIN_PASSWORD` as a secret environment
variable with at least 12 characters; never commit it. Use HTTPS for hosted access.

The dashboard has 1/7/30/90-day filters, anonymous browser session counts, successful
prediction request counts, records analyzed, predicted failure alerts, failed actions,
activity charts, recent activity, usage CSV export and private prediction history.
Downloads tracked: input template, batch prediction CSV and research report.
Counts start with this version; historical prediction rows remain available privately
but cannot be attributed to past visits. Browser sessions are not unique people or
concurrent users. API requests have no visitor identity. Chart dates are UTC.
Analytics stores action metadata plus the visitor's IP address and reported browser/device details for new dashboard activity. These details are visible only in the authenticated admin dashboard and its export. The public dashboard displays a collection notice. Uploaded filenames are not recorded; submitted sensor readings remain in the separate private prediction history.

Cloudflare Quick Tunnel visits use the reported `CF-Connecting-IP` header, with its source labeled in the activity log. Direct local visits may show a loopback address. Device and browser labels are inferred from the user agent and can be inaccurate or spoofed; IP addresses do not uniquely identify people. Historical events retain blank visitor fields. Restart Streamlit after updating Python modules so the running app loads the new code.
Prediction storage separately retains submitted readings and model results for the admin.

`GET /history` and `GET /admin/usage?days=30` now require
`Authorization: Bearer <your-admin-password>`. Set the same secret in the API process.
Missing or incorrect credentials return 401, including when no password is configured.
The public dashboard no longer displays shared prediction history. Public prediction
endpoints remain available without login. This is a small prototype: before opening a
busy public service, use gateway rate limiting and managed administrator authentication.
The login retry delay applies only within a browser session, not across clients.

SQLite persists only on the configured disk. Set `MAINTENANCE_DB` to a database path
on persistent storage for hosting; ephemeral hosting can lose counts and history.
There is no automatic retention purge. The admin view refreshes on interaction or via
**Refresh dashboard**, not through background polling. No sample usage is fabricated.
All **23 automated tests** cover model behavior, admin access, session counting,
API activity, invalid requests and dashboard login/logout.
