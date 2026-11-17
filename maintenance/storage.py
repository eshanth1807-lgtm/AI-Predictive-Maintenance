"""Each operation owns its connection; shared by Streamlit and FastAPI."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from maintenance.config import ROOT


def connect():
    path = Path(os.environ.get('MAINTENANCE_DB', str(ROOT / 'data' / 'predictions.sqlite3')))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
        input_json TEXT NOT NULL, output_json TEXT NOT NULL)''')
    connection.execute("""CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL, action TEXT NOT NULL, source TEXT NOT NULL,
        session TEXT NOT NULL, records INTEGER NOT NULL, failures INTEGER NOT NULL, status TEXT NOT NULL)""")
    connection.execute('CREATE INDEX IF NOT EXISTS events_created_at ON events(created_at)')
    return connection


def save_many(records, results):
    connection = connect()
    try:
        with connection:
            for record, result in zip(records, results, strict=True):
                cursor = connection.execute(
                    'INSERT INTO predictions(created_at,input_json,output_json) VALUES(?,?,?)',
                    (datetime.now(timezone.utc).isoformat(), json.dumps(record.model_dump()), json.dumps(result)))
                result['history_id'] = cursor.lastrowid
    finally:
        connection.close()


def history(limit=100):
    if not 1 <= limit <= 1000:
        raise ValueError('limit must be between 1 and 1000')
    connection = connect()
    try:
        rows = connection.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(id=row[0], created_at=row[1], inputs=json.loads(row[2]), result=json.loads(row[3])) for row in rows]
    finally:
        connection.close()
