import sqlite3
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from maintenance.admin import valid_password
from maintenance.analytics import event, summary
from maintenance.config import NOTICE
from maintenance.schema import MachineInput, BatchInput
from maintenance.predictor import load_bundle, predict_many
from maintenance.storage import history

app = FastAPI(title='Industrial Equipment Failure Detection', version='1.0.0', description=NOTICE)


bearer = HTTPBearer(auto_error=False)
def require_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
    if credentials is None or not valid_password(credentials.credentials):
        raise HTTPException(401, 'Administrator authentication required.', headers={'WWW-Authenticate': 'Bearer'})


@app.get('/admin/usage', dependencies=[Depends(require_admin)])
def usage(days: int = Query(30)):
    if days not in (1, 7, 30, 90):
        raise HTTPException(422, 'Unsupported date range')
    try:
        return {'events': summary(days)}
    except sqlite3.Error as exc:
        raise HTTPException(503, 'Activity database unavailable.') from exc


@app.middleware('http')
async def record_prediction_requests(request, call_next):
    response = await call_next(request)
    if request.method == 'POST' and request.url.path in ('/predict', '/predict/batch') and response.status_code >= 400:
        event('single_prediction' if request.url.path == '/predict' else 'batch_prediction', source='api', status='error')
    return response


@app.get('/health')
def health():
    try:
        bundle = load_bundle()
        return {'status': 'ok', 'model_version': bundle['version']}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


def run_prediction(records, action='single_prediction'):
    try:
        results = predict_many(records)
        event(action, source='api', records=len(results), failures=sum(r['prediction'] for r in results))
        return results
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(503, 'Prediction history database unavailable.') from exc


@app.post('/predict')
def predict(record: MachineInput):
    return run_prediction([record])[0]


@app.post('/predict/batch')
def predict_batch(batch: BatchInput):
    return {'predictions': run_prediction(batch.records, 'batch_prediction')}


@app.get('/history', dependencies=[Depends(require_admin)])
def get_history(limit: int = Query(100, ge=1, le=1000)):
    try:
        return {'predictions': history(limit)}
    except sqlite3.Error as exc:
        raise HTTPException(503, 'Prediction history database unavailable.') from exc


@app.get('/model')
def model_info():
    try:
        bundle = load_bundle()
        return {key: bundle[key] for key in ('version', 'features', 'threshold', 'training_ranges', 'selected_model')}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
