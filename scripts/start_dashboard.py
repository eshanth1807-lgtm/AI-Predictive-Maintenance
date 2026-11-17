"""Start the dashboard with a private password, without saving it in source files."""
import getpass
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
password = os.environ.get('MAINTENANCE_ADMIN_PASSWORD', '')
while len(password) < 12:
    password = getpass.getpass('Choose admin password (at least 12 characters; hidden): ')
    if len(password) < 12:
        print('Use at least 12 characters.')
environment = os.environ.copy()
environment['MAINTENANCE_ADMIN_PASSWORD'] = password
raise SystemExit(subprocess.call([sys.executable, '-m', 'streamlit', 'run', str(root / 'app.py'),
                                 '--server.address', '127.0.0.1'], cwd=root, env=environment))
