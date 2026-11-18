"""Private activity metadata. Never store cookies or authorization headers."""
import ipaddress
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from maintenance.storage import connect


def visitor_details(headers, peer=None):
    headers = {k.lower(): v for k, v in headers.items()}
    ua = headers.get('user-agent', '')[:512]
    ip, origin = peer or '', 'Connection'
    # Cloudflare overwrites this header on tunnel traffic. Keep provenance visible.
    host = headers.get('host', '').split(':')[0].lower()
    if host.endswith('.trycloudflare.com') and headers.get('cf-connecting-ip'):
        ip, origin = headers['cf-connecting-ip'], 'Cloudflare header (reported)'
    try:
        ip = str(ipaddress.ip_address(ip))
    except ValueError:
        ip, origin = '', 'Unavailable'
    device = next((label for token, label in [('iPad', 'iPad'), ('iPhone', 'iPhone'), ('Android', 'Android'), ('Windows', 'Windows computer'), ('Macintosh', 'Mac computer'), ('Linux', 'Linux computer')] if token in ua), 'Unknown')
    browser = next((label for token, label in [('Edg/', 'Edge'), ('OPR/', 'Opera'), ('Firefox/', 'Firefox'), ('FxiOS/', 'Firefox'), ('Chrome/', 'Chrome'), ('CriOS/', 'Chrome'), ('Safari/', 'Safari')] if token in ua), 'Unknown')
    return dict(ip_address=ip, ip_source=origin, device=device, browser=browser, user_agent=ua)


def event(action, source='dashboard', session='', records=0, failures=0, status='ok', visitor=None):
    try:
        db = connect()
        try:
            with db:
                visitor = visitor or {}
                db.execute('INSERT INTO events(created_at,action,source,session,records,failures,status,ip_address,ip_source,device,browser,user_agent) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                           (datetime.now(timezone.utc).isoformat(), action, source, session, records, failures, status,
                            *(visitor.get(key, '') for key in ('ip_address', 'ip_source', 'device', 'browser', 'user_agent'))))
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
