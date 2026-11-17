from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
TARGET = 'Machine failure'
FORBIDDEN = ['UDI', 'Product ID', TARGET, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
MODEL_PATH = ROOT / 'model' / 'classifier.joblib'
METRICS_PATH = ROOT / 'reports' / 'metrics.json'
DATA_PATH = ROOT / 'data' / 'ai4i2020.csv'
SEED = 42
NOTICE = ('Research prototype trained on synthetic AI4I data. Classifies failure at the '
          'recorded condition; does not estimate time to failure or establish equipment safety.')
