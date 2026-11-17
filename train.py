"""Reproducible experiment: fit on train, select on validation, evaluate once on test."""
import hashlib
import json
import platform
from datetime import datetime, timezone
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score,
                             roc_auc_score, ConfusionMatrixDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from maintenance.config import ROOT, DATA_PATH, FEATURES, TARGET, SEED


def split_indices(data):
    train, remainder = train_test_split(np.arange(len(data)), test_size=0.4,
                                        stratify=data[TARGET], random_state=SEED)
    validation, test = train_test_split(remainder, test_size=0.5,
                                       stratify=data.iloc[remainder][TARGET], random_state=SEED)
    return train, validation, test


def evaluate(y, scores, threshold):
    predictions = scores >= threshold
    return dict(accuracy=float(accuracy_score(y, predictions)),
                precision=float(precision_score(y, predictions, zero_division=0)),
                recall=float(recall_score(y, predictions, zero_division=0)),
                f1=float(f1_score(y, predictions, zero_division=0)),
                f2=float(fbeta_score(y, predictions, beta=2, zero_division=0)),
                average_precision=float(average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)),
                confusion_matrix=confusion_matrix(y, predictions, labels=[0, 1]).tolist())


def choose_threshold(y, scores):
    # Fixed grid and F2 objective chosen before accessing test labels.
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    return float(max(thresholds, key=lambda t: (fbeta_score(y, scores >= t, beta=2, zero_division=0), t)))


def main():
    if not DATA_PATH.exists():
        raise SystemExit('Missing data. Run: python scripts/download_data.py')
    data = pd.read_csv(DATA_PATH)
    if len(data) != 10000 or data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError('Expected 10,000 complete AI4I records.')
    if set(data[TARGET].unique()) != {0, 1} or not set(data.Type).issubset({'L', 'M', 'H'}):
        raise ValueError('Unexpected labels or machine types.')
    train, validation, test = split_indices(data)
    x_train, y_train = data.iloc[train][FEATURES], data.iloc[train][TARGET]
    x_val, y_val = data.iloc[validation][FEATURES], data.iloc[validation][TARGET]
    x_test, y_test = data.iloc[test][FEATURES], data.iloc[test][TARGET]
    classifiers = {
        'Dummy baseline': DummyClassifier(strategy='prior'),
        'Logistic regression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED),
        'Decision tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=SEED),
        'Random forest': RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                               class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
    }
    pipelines, thresholds, validation_metrics = {}, {}, {}
    for name, classifier in classifiers.items():
        preprocessor = ColumnTransformer([
            ('type', OneHotEncoder(handle_unknown='error', sparse_output=False), ['Type']),
            ('numeric', StandardScaler(), FEATURES[1:])], remainder='drop')
        pipeline = Pipeline([('preprocess', preprocessor), ('classifier', classifier)])
        pipeline.fit(x_train, y_train)
        scores = pipeline.predict_proba(x_val)[:, 1]
        threshold = 0.5 if name == 'Dummy baseline' else choose_threshold(y_val, scores)
        pipelines[name], thresholds[name] = pipeline, threshold
        validation_metrics[name] = evaluate(y_val, scores, threshold)
    selected = max([n for n in classifiers if n != 'Dummy baseline'],
                   key=lambda n: validation_metrics[n]['f2'])
    # No tuning, retraining, or model selection below this line.
    test_metrics = {name: evaluate(y_test, pipeline.predict_proba(x_test)[:, 1], thresholds[name])
                    for name, pipeline in pipelines.items()}
    selected_pipeline, threshold = pipelines[selected], thresholds[selected]
    scores = selected_pipeline.predict_proba(x_test)[:, 1]
    predictions = (scores >= threshold).astype(int)
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    (ROOT / 'model').mkdir(exist_ok=True)
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    version = hashlib.sha256((digest + selected + str(threshold) + sklearn.__version__ + 'v1').encode()).hexdigest()[:12]
    ranges = {name: [float(x_train[name].min()), float(x_train[name].max())] for name in FEATURES[1:]}
    joblib.dump(dict(pipeline=selected_pipeline, features=FEATURES, threshold=threshold,
                     selected_model=selected, training_ranges=ranges, version=version),
                ROOT / 'model' / 'classifier.joblib', compress=3)
    split_frame = pd.DataFrame({'row_index': np.arange(len(data)), 'split': ''})
    for name, indices in [('train', train), ('validation', validation), ('test', test)]:
        split_frame.loc[indices, 'split'] = name
    split_frame.to_csv(reports / 'split_assignments.csv', index=False)
    errors = data.iloc[test].copy()
    errors['failure_score'], errors['prediction'] = scores, predictions
    errors.to_csv(reports / 'test_predictions.csv', index=False)
    false_negatives = errors[(errors[TARGET] == 1) & (errors.prediction == 0)]
    false_negatives.to_csv(reports / 'false_negatives.csv', index=False)
    rng = np.random.default_rng(SEED)
    recalls = []
    positives = predictions[y_test.to_numpy() == 1]
    for _ in range(2000):
        recalls.append(float(rng.choice(positives, len(positives), replace=True).mean()))
    metrics = dict(selected_model=selected, threshold=threshold, model_version=version,
        created_at=datetime.now(timezone.utc).isoformat(), dataset_sha256=digest, seed=SEED,
        python_version=platform.python_version(), sklearn_version=sklearn.__version__,
        features=FEATURES, rows=len(data), failures=int(data[TARGET].sum()),
        split_sizes={'train': len(train), 'validation': len(validation), 'test': len(test)},
        split_failures={'train': int(y_train.sum()), 'validation': int(y_val.sum()), 'test': int(y_test.sum())},
        selection='Highest validation F2; threshold grid 0.05 to 0.95, step 0.01; ties prefer higher threshold.',
        validation=validation_metrics, test=test_metrics, thresholds=thresholds,
        recall_bootstrap_95_ci=np.quantile(recalls, [0.025, 0.975]).tolist(),
        false_negative_modes={mode: int(false_negatives[mode].sum()) for mode in ['TWF','HDF','PWF','OSF','RNF']},
        training_ranges=ranges)
    (reports / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=['No failure', 'Failure'],
                                           cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(f'{selected}: held-out test set')
    fig.tight_layout(); fig.savefig(reports / 'confusion_matrix.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pipeline in pipelines.items():
        PrecisionRecallDisplay.from_predictions(y_test, pipeline.predict_proba(x_test)[:, 1], name=name, ax=ax)
    ax.set_title('Precision-recall comparison | test set')
    fig.tight_layout(); fig.savefig(reports / 'precision_recall.png'); plt.close(fig)
    importance = permutation_importance(selected_pipeline, x_test, y_test, scoring='average_precision',
                                        n_repeats=10, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({'feature': FEATURES, 'importance': importance.importances_mean,
                        'std': importance.importances_std}).sort_values('importance')
    imp.to_csv(reports / 'feature_importance.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.feature, imp.importance, xerr=imp['std'], color='#14b8a6')
    ax.set_xlabel('Decrease in test average precision after permutation')
    ax.set_title('Global feature importance (not a causal explanation)')
    fig.tight_layout(); fig.savefig(reports / 'feature_importance.png'); plt.close(fig)
    print(json.dumps({'selected_model': selected, 'threshold': threshold, 'test': test_metrics[selected]}, indent=2))
    from scripts.build_report import build_report
    build_report()


if __name__ == '__main__':
    main()
