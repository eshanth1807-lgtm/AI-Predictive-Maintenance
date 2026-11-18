import uuid
import json
import sqlite3
import pandas as pd
import streamlit as st
from pydantic import ValidationError
from maintenance.config import ROOT, METRICS_PATH, NOTICE
from maintenance.schema import MachineInput
from maintenance.predictor import load_bundle, predict_many
from maintenance.analytics import event, visitor_details
from maintenance.admin import render as render_admin, is_authenticated

st.set_page_config(page_title='MachineWatch | Failure Detection', page_icon='⚙️', layout='wide')
page = 'Machine dashboard'
if is_authenticated():
    page = st.sidebar.radio('Navigation', ['Machine dashboard', 'Admin dashboard'],
                            index=1 if st.query_params.get('admin') == '1' else 0, key='admin_navigation')
elif st.query_params.get('admin') == '1':
    render_admin()
    st.stop()
if page == 'Admin dashboard':
    render_admin()
    st.stop()
visitor = visitor_details(st.context.headers, st.context.ip_address)
if 'visitor_session' not in st.session_state:
    st.session_state['visitor_session'] = uuid.uuid4().hex
    event('visit', session=st.session_state['visitor_session'], visitor=visitor)
def track(action, **kwargs):
    event(action, session=st.session_state['visitor_session'], visitor=visitor, **kwargs)
st.caption('Usage notice: this app stores your IP address, reported browser/device details, feature activity and submitted readings privately for the administrator. Uploaded filenames are not recorded.')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1250px;}
[data-testid="stMetric"] {background:#102937; border:1px solid #244351; border-radius:12px; padding:18px;}
h1 {letter-spacing:-1.4px;} .eyebrow {color:#36d6b0;font-size:12px;letter-spacing:3px;}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">MACHINEWATCH / INDUSTRIAL INTELLIGENCE</div>', unsafe_allow_html=True)
st.title('Understand the operating condition.')
st.caption('AI-based industrial equipment failure detection • Research prototype')
try:
    bundle = load_bundle()
    metrics = json.loads(METRICS_PATH.read_text())
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.code('python scripts/download_data.py\npython train.py')
    st.stop()

with st.sidebar:
    st.title('⚙️ MachineWatch')
    st.caption('LOCAL WORKSPACE')
    st.success('Model ready')
    st.write(bundle['selected_model'])
    st.caption(f'Model version: {bundle["version"]}')
    st.divider()
    st.write('Six operating measurements. One consistent prediction pipeline.')
    st.caption(NOTICE)
    st.caption('Temperature: Kelvin • Speed: rpm • Torque: Nm • Wear: minutes')

test = metrics['test'][metrics['selected_model']]
cols = st.columns(4)
for col, label, value in zip(cols, ['Dataset records', 'Test failure recall', 'Test precision', 'Decision threshold'],
                            [f'{metrics["rows"]:,}', f'{test["recall"]:.1%}', f'{test["precision"]:.1%}', f'{bundle["threshold"]:.2f}']):
    col.metric(label, value)
st.metric('Test accuracy', f'{test["accuracy"]:.2%}')
st.caption('Evaluation update: this existing synthetic test set has been inspected in earlier versions; confirm improvements on new independent data before deployment decisions.')
st.write('')
single, batch, evaluation = st.tabs(['Condition check', 'Batch analysis', 'Model evaluation'])
with single:
    left, right = st.columns([1.2, 1], gap='large')
    with left:
        st.subheader('Machine operating inputs')
        st.caption('Enter one snapshot of machine conditions. Inputs are checked before inference.')
        with st.form('prediction'):
            a, b = st.columns(2)
            machine_type = a.selectbox('Machine type', ['M','L','H'])
            air = b.number_input('Air temperature (K)', min_value=250.0, max_value=400.0, value=300.5, step=0.1)
            process = a.number_input('Process temperature (K)', min_value=250.0, max_value=500.0, value=310.2, step=0.1)
            speed = b.number_input('Rotational speed (rpm)', min_value=1, max_value=10000, value=1500)
            torque = a.number_input('Torque (Nm)', min_value=0.0, max_value=200.0, value=42.0, step=0.1)
            wear = b.number_input('Tool wear (min)', min_value=0, max_value=1000, value=120)
            submitted = st.form_submit_button('Analyze condition', type='primary', width='stretch')
        if submitted:
            st.session_state.pop('result', None)
            try:
                record = MachineInput(machine_type=machine_type, air_temperature=air, process_temperature=process,
                                      rotational_speed=speed, torque=torque, tool_wear=wear)
                st.session_state['result'] = predict_many([record])[0]
                track('single_prediction', records=1, failures=st.session_state['result']['prediction'])
            except (ValidationError, ValueError, sqlite3.Error) as exc:
                track('single_prediction', status='error')
                st.error(f'Could not complete prediction: {exc}')
    with right:
        st.subheader('Condition assessment')
        if 'result' not in st.session_state:
            st.info('Enter readings and select Analyze condition to see the model assessment.')
            st.caption('No prediction has been made yet. The result is saved to local history only when you submit.')
        else:
            result = st.session_state['result']
            if result['prediction']:
                st.error('Failure predicted — review the operating condition.')
            else:
                st.success('No failure predicted for these readings.')
            st.metric('Model failure score', f'{result["failure_score"]:.3f}')
            st.progress(result['failure_score'])
            st.caption(f'Failure is predicted at scores ≥ {result["threshold"]:.2f}. This score is not a calibrated real-world probability.')
            for warning in result['warnings']:
                st.warning(warning)
            st.caption('A low score does not prove the equipment is safe. Follow established inspection and maintenance procedures.')
    st.divider()
    st.caption(NOTICE)

with batch:
    st.subheader('Analyze a CSV of operating conditions')
    st.write('Upload up to 1,000 rows using the six columns in the sample. Extra columns and invalid rows are rejected before any prediction is saved.')
    sample = pd.DataFrame([MachineInput().model_dump()])
    st.download_button('Download input template', sample.to_csv(index=False), 'machine_inputs.csv', 'text/csv', on_click=track, args=('template_download',))
    uploaded = st.file_uploader('Input CSV (maximum 2 MB)', type=['csv'])
    if uploaded is not None and st.button('Analyze batch', type='primary'):
        st.session_state.pop('batch_output', None)
        try:
            if uploaded.size > 2_000_000:
                raise ValueError('File exceeds the 2 MB limit.')
            frame = pd.read_csv(uploaded)
            expected = set(MachineInput.model_fields)
            if set(frame.columns) != expected:
                raise ValueError('CSV columns must exactly match the input template.')
            if not 1 <= len(frame) <= 1000:
                raise ValueError('Provide between 1 and 1,000 rows.')
            records = []
            for number, row in enumerate(frame.to_dict('records'), start=2):
                try:
                    records.append(MachineInput(**row))
                except ValidationError as exc:
                    raise ValueError(f'CSV line {number}: {exc}') from exc
            results = predict_many(records)
            track('batch_prediction', records=len(results), failures=sum(r['prediction'] for r in results))
            output = pd.concat([frame.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            output['warnings'] = output.warnings.apply('; '.join)
            st.session_state['batch_output'] = output
        except (ValueError, pd.errors.ParserError, sqlite3.Error) as exc:
            track('batch_prediction', status='error')
            st.error(str(exc))
    if 'batch_output' in st.session_state:
        output = st.session_state['batch_output']
        st.info(f'{len(output)} records analyzed • {int(output.prediction.sum())} failure alerts • {len(output) - int(output.prediction.sum())} without a predicted failure')
        st.dataframe(output, width='stretch')
        st.download_button('Download predictions', output.to_csv(index=False), 'predictions.csv', 'text/csv', on_click=track, args=('predictions_download',))

with evaluation:
    st.subheader('Evidence behind the model')
    tn, fp, fn, tp = sum(test['confusion_matrix'], [])
    st.write(f'On 2,000 test records: {tp} failures detected, {fn} failures missed, and {fp} false alarms.')
    st.write('The new candidate combines temperature difference, mechanical power (torque × angular speed), and wear × torque with the original six inputs. These describe operating relationships; they are not proof of the cause of a failure.')
    st.caption('6,000 train / 2,000 validation / 2,000 test. Model and threshold chosen only on validation F2.')
    comparison = pd.DataFrame(metrics['test']).T[['accuracy','precision','recall','f1','f2','average_precision','roc_auc']]
    st.dataframe(comparison.style.format('{:.3f}'), width='stretch')
    a, b = st.columns(2)
    a.image(str(ROOT / 'reports/confusion_matrix.png'), width='stretch')
    b.image(str(ROOT / 'reports/precision_recall.png'), width='stretch')
    st.image(str(ROOT / 'reports/feature_importance.png'), width=800)
    st.caption('Permutation importance describes global associations, not causal effects or a reason for a specific prediction.')
    st.warning('Synthetic data and random splits limit generalization. This is not a real-world safety validation or time-to-failure model.')
    report = ROOT / 'research_report.pdf'
    if report.exists():
        st.download_button('Download research report', report.read_bytes(), report.name, 'application/pdf', on_click=track, args=('report_download',))
