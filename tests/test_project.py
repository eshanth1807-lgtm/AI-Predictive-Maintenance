import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from api import app
from maintenance.config import ROOT, DATA_PATH, FEATURES, FORBIDDEN
from maintenance.schema import MachineInput
from maintenance.predictor import load_bundle, predict_many
from maintenance.storage import history
from train import split_indices, choose_threshold


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setenv('MAINTENANCE_ADMIN_PASSWORD', 'test-admin-password-only')
    monkeypatch.setenv('MAINTENANCE_DB', str(tmp_path / 'history.sqlite3'))


@pytest.mark.parametrize('field,value', [('machine_type','X'),('air_temperature',float('nan')),
    ('torque',float('inf')),('tool_wear',-1),('rotational_speed',0),('rotational_speed',1.5)])
def test_invalid_inputs(field, value):
    with pytest.raises(ValidationError):
        MachineInput(**{field: value})


def test_reject_leakage_fields():
    with pytest.raises(ValidationError):
        MachineInput(TWF=1)
    assert not set(FEATURES).intersection(FORBIDDEN)


def test_split_is_disjoint_complete_and_stratified():
    data = pd.read_csv(DATA_PATH)
    splits = split_indices(data)
    assert [len(s) for s in splits] == [6000,2000,2000]
    assert len(set(np.concatenate(splits))) == len(data)
    for indices in splits:
        assert abs(data.iloc[indices]['Machine failure'].mean() - data['Machine failure'].mean()) < .001


def test_preprocessor_fits_training_only():
    data = pd.read_csv(DATA_PATH)
    indices, _, _ = split_indices(data)
    scaler = load_bundle()['pipeline'].named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(scaler.mean_, data.iloc[indices][FEATURES[1:]].mean())


def test_threshold_selection_uses_f2():
    assert choose_threshold(np.array([0,0,1,1]), np.array([.1,.2,.3,.4])) == .30


def test_saved_metrics_match_model():
    data = pd.read_csv(DATA_PATH)
    _, _, indices = split_indices(data)
    bundle = load_bundle()
    scores = bundle['pipeline'].predict_proba(data.iloc[indices][FEATURES])[:, 1]
    saved = pd.read_csv(ROOT / 'reports/test_predictions.csv')
    np.testing.assert_allclose(scores, saved.failure_score)
    np.testing.assert_array_equal(scores >= bundle['threshold'], saved.prediction)


def test_predictions_and_history():
    result = predict_many([MachineInput()])[0]
    assert 0 <= result['failure_score'] <= 1
    assert result['prediction'] == int(result['failure_score'] >= result['threshold'])
    assert result['history_id'] == history()[0]['id']
    assert history()[0]['inputs']['torque'] == 42


def test_out_of_distribution_warning():
    result = predict_many([MachineInput(air_temperature=390)], persist=False)[0]
    assert any('Air temperature' in w for w in result['warnings'])
    assert history() == []


def test_api_single_batch_parity():
    client = TestClient(app)
    record = MachineInput().model_dump()
    assert client.get('/health').status_code == 200
    assert client.get('/model').status_code == 200
    single = client.post('/predict', json=record)
    assert single.status_code == 200
    batch = client.post('/predict/batch', json={'records':[record, record]})
    assert batch.status_code == 200
    assert single.json()['failure_score'] == batch.json()['predictions'][0]['failure_score']
    assert len(client.get('/history', headers={'Authorization': 'Bearer test-admin-password-only'}).json()['predictions']) == 3


def test_api_invalid_batch_is_atomic():
    client = TestClient(app)
    response = client.post('/predict/batch', json={'records':[{}, {'torque':-1}]})
    assert response.status_code == 422
    assert history() == []
    assert client.post('/predict/batch', json={'records':[]}).status_code == 422
    assert client.post('/predict/batch', json={'records':[{}]*1001}).status_code == 422
    assert client.get('/history?limit=1001', headers={'Authorization': 'Bearer test-admin-password-only'}).status_code == 422


def test_missing_model_gives_actionable_error(monkeypatch):
    import api
    def missing():
        raise FileNotFoundError('Model missing. Run: python train.py')
    monkeypatch.setattr(api, 'load_bundle', missing)
    response = TestClient(app).get('/health')
    assert response.status_code == 503
    assert 'train.py' in response.json()['detail']


def test_streamlit_single_and_batch():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert not at.exception
    submit = next(b for b in at.button if b.label == 'Analyze condition')
    submit.click().run(timeout=30)
    assert not at.exception
    assert len(history()) == 1
    assert any(m.label == 'Model failure score' for m in at.metric)


def test_api_history_database_failure(monkeypatch):
    import api
    import sqlite3
    def unavailable(limit):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(api, 'history', unavailable)
    response = TestClient(app).get('/history', headers={'Authorization': 'Bearer test-admin-password-only'})
    assert response.status_code == 503
    assert response.json()['detail'] == 'Prediction history database unavailable.'


def test_api_model_schema_failure(monkeypatch):
    import api
    def mismatch(*args):
        raise ValueError('Model feature schema mismatch. Retrain the model.')
    monkeypatch.setattr(api, 'load_bundle', mismatch)
    monkeypatch.setattr(api, 'predict_many', mismatch)
    client = TestClient(app)
    for path in ('/health', '/model'):
        assert client.get(path).status_code == 503
    assert client.post('/predict', json={}).status_code == 503
    assert client.post('/predict/batch', json={'records': [{}]}).status_code == 503


def test_dashboard_failed_prediction_clears_previous_result(monkeypatch):
    import maintenance.predictor
    import sqlite3
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    next(b for b in at.button if b.label == 'Analyze condition').click().run(timeout=30)
    assert any(m.label == 'Model failure score' for m in at.metric)
    def unavailable(*args):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(maintenance.predictor, 'predict_many', unavailable)
    next(b for b in at.button if b.label == 'Analyze condition').click().run(timeout=30)
    assert not at.exception
    assert not any(m.label == 'Model failure score' for m in at.metric)
    assert any('Could not complete prediction' in e.value for e in at.error)


def test_admin_api_protection_and_tracking(monkeypatch):
    from maintenance.analytics import summary
    client = TestClient(app)
    for path in ['/history', '/admin/usage']:
        assert client.get(path).status_code == 401
        assert client.get(path, headers={'Authorization': 'Bearer wrong'}).status_code == 401
    record = MachineInput().model_dump()
    client.post('/predict', json=record)
    client.post('/predict/batch', json={'records': [record, record]})
    client.post('/predict', json={'torque': -1})
    events = summary()
    assert len(events) == 3
    assert sum(e['records'] for e in events) == 3
    assert sum(e['status'] == 'error' for e in events) == 1
    assert all(e['session'] == '' for e in events)
    headers = {'Authorization': 'Bearer test-admin-password-only'}
    assert client.get('/admin/usage', headers=headers).status_code == 200
    assert client.get('/admin/usage?days=2', headers=headers).status_code == 422
    monkeypatch.delenv('MAINTENANCE_ADMIN_PASSWORD')
    assert client.get('/history', headers=headers).status_code == 401


def test_admin_dashboard_login_and_logout():
    from streamlit.testing.v1 import AppTest
    from maintenance.analytics import summary
    at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert not at.exception
    at.run()
    assert len([e for e in summary() if e['action'] == 'visit']) == 1
    assert not at.radio
    at.query_params['admin'] = '1'
    at.run()
    assert not at.exception
    assert not at.dataframe
    at.text_input[0].set_value('test-admin-password-only')
    next(b for b in at.button if b.label == 'Sign in').click().run()
    assert not at.exception
    assert any(m.label == 'Browser sessions' for m in at.metric)
    next(b for b in at.button if b.label == 'Sign out').click().run()
    assert not at.exception
    assert not at.radio
    assert not at.text_input
    assert 'admin' not in at.query_params


def test_admin_menu_hidden_after_session_expiry():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'))
    at.query_params['admin'] = '1'
    at.run(timeout=30)
    assert not at.radio
    assert not at.dataframe
    at.text_input[0].set_value('test-admin-password-only')
    next(b for b in at.button if b.label == 'Sign in').click().run()
    assert len(at.radio) == 1
    at.session_state['admin_until'] = 0
    at.run()
    assert not at.exception
    assert not at.radio
    assert not at.dataframe
    assert len(at.text_input) == 1
