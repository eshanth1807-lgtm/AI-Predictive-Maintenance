"""Fail-closed administrator access shared by Streamlit and the API."""
import sqlite3
import hashlib
import hmac
import os
import time


def password():
    value = os.environ.get('MAINTENANCE_ADMIN_PASSWORD', '')
    return value if len(value) >= 12 else ''


def valid_password(value):
    expected = password()
    return bool(expected) and hmac.compare_digest(value.encode(), expected.encode())


def is_authenticated():
    import streamlit as st
    expected = password()
    return bool(expected) and (
        st.session_state.get('admin_fingerprint') == hashlib.sha256(expected.encode()).hexdigest()
        and time.time() < st.session_state.get('admin_until', 0))


def render():
    import pandas as pd
    import streamlit as st
    from maintenance.analytics import summary
    from maintenance.storage import history
    st.title('Admin dashboard')
    st.caption('Private usage overview • All timestamps are UTC')
    if not password():
        st.info('Admin access is disabled. Set MAINTENANCE_ADMIN_PASSWORD to a unique password of at least 12 characters before starting the app.')
        return
    fingerprint = hashlib.sha256(password().encode()).hexdigest()
    authenticated = is_authenticated()
    if not authenticated:
        with st.form('admin_login'):
            entered = st.text_input('Admin password', type='password')
            login = st.form_submit_button('Sign in')
        if login:
            if time.time() < st.session_state.get('admin_retry_after', 0):
                st.error('Please wait before trying again.')
            elif valid_password(entered):
                st.session_state['admin_fingerprint'] = fingerprint
                st.session_state['admin_until'] = time.time() + 1800
                st.rerun()
            else:
                st.session_state['admin_retry_after'] = time.time() + 5
                st.error('Incorrect password.')
        return
    if st.button('Sign out'):
        st.session_state.pop('admin_fingerprint', None)
        st.session_state.pop('admin_until', None)
        st.session_state.pop('admin_navigation', None)
        st.query_params.pop('admin', None)
        st.rerun()
    days = st.selectbox('Period', [1, 7, 30, 90], index=2, format_func=lambda n: f'Last {n} days')
    st.button('Refresh dashboard')
    st.caption('Sessions are anonymous browser sessions, not unique people. Reloads may start a new session. API requests are counted separately. Tracking starts with this version.')
    try:
        frame = pd.DataFrame(summary(days))
        if frame.empty:
            st.info('No activity recorded in this period yet.')
        else:
            visits = frame[frame.action == 'visit']
            predictions = frame[(frame.action.isin(['single_prediction', 'batch_prediction'])) & (frame.status == 'ok')]
            cols = st.columns(4)
            for col, label, value in zip(cols, ['Browser sessions', 'Prediction requests', 'Records analyzed', 'Failure alerts'],
                    [visits.session.nunique(), len(predictions), int(predictions.records.sum()), int(predictions.failures.sum())]):
                col.metric(label, value)
            st.metric('Failed actions', int((frame.status == 'error').sum()))
            a, b = st.columns(2)
            a.subheader('Activity by day')
            a.bar_chart(frame.assign(day=frame.created_at.str[:10]).groupby('day').size())
            b.subheader('Features used')
            b.bar_chart(frame.groupby('action').size())
            st.subheader('Recent activity')
            st.caption('IP address and device details are recorded for new activity only. Device/browser labels are inferred from the reported user agent and may be inaccurate or spoofed. Shared networks and VPNs can share an IP; these fields do not identify a person. Local access may show a loopback address. Older records have blank details.')
            public = frame.drop(columns=['session', 'id'])
            st.dataframe(public.head(200), width='stretch', hide_index=True)
            st.download_button('Export usage CSV', public.to_csv(index=False), 'usage.csv', 'text/csv')
        with st.expander('Private prediction history — latest 100 records'):
            records = history()
            if records:
                st.dataframe(pd.json_normalize(records), width='stretch')
            else:
                st.info('No predictions saved.')
    except (OSError, sqlite3.Error):
        st.error('Activity database unavailable. Check the configured database path and permissions.')
    st.caption('SQLite needs a persistent disk for hosted use. This dashboard does not identify a visitor or infer their job or purpose.')
