"""Aggregate activity metadata; never store IP addresses, filenames or input values."""
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from maintenance.storage import connect


def event(action, source='dashboard', session='', records=0, failures=0, status='ok'):
    try:
        db = connect()
        try:
            with db:
                db.execute('INSERT INTO events(created_at,action,source,session,records,failures,status) VALUES(?,?,?,?,?,?,?)',
                           (datetime.now(timezone.utc).isoformat(), action, source, session, records, failures, status))
        finally:
            db.close()
    except (OSError, sqlite3.Error) as exc:
        # Analytics must not prevent a successful prediction. Do not log user inputs.
        logging.getLogger(__name__).warning('Activity recording unavailable: %s', type(exc).__name__)


def summary(days=30):
    if days not in (1, 7, 30, 90):
        raise ValueError('Unsupported date range')
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    db = connect()
    try:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM events WHERE created_at >= ? ORDER BY id DESC', (since,))]
        return rows
    finally:
        db.close()
