from functools import lru_cache
import joblib
import pandas as pd
from maintenance.config import FEATURES, MODEL_PATH, NOTICE
from maintenance.storage import save_many


@lru_cache(maxsize=1)
def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('Model missing. Run: python train.py')
    # Only load the locally generated, trusted artifact. Never accept uploaded pickle/joblib files.
    bundle = joblib.load(MODEL_PATH)
    if bundle['features'] != FEATURES:
        raise ValueError('Model feature schema mismatch. Retrain the model.')
    return bundle


def predict_many(records, persist=True):
    bundle = load_bundle()
    frame = pd.DataFrame([r.feature_row() for r in records], columns=FEATURES)
    scores = bundle['pipeline'].predict_proba(frame)[:, 1]
    results = []
    for row, score in zip(frame.to_dict('records'), scores, strict=True):
        warnings = [f'{name} is outside the training range ({bounds[0]:g} to {bounds[1]:g}).'
                    for name, bounds in bundle['training_ranges'].items()
                    if not bounds[0] <= row[name] <= bounds[1]]
        if row['Process temperature [K]'] < row['Air temperature [K]']:
            warnings.append('Process temperature is below air temperature; verify the readings.')
        label = int(score >= bundle['threshold'])
        results.append(dict(prediction=label, condition='Failure predicted' if label else 'No failure predicted',
                            failure_score=float(score), threshold=bundle['threshold'],
                            model_version=bundle['version'], warnings=warnings, notice=NOTICE))
    if persist:
        save_many(records, results)
    return results
