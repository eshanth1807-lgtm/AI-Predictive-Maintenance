import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def build_report():
    m = json.loads((ROOT / 'reports/metrics.json').read_text())
    s = m['test'][m['selected_model']]
    tn, fp, fn, tp = sum(s['confusion_matrix'], [])
    styles = getSampleStyleSheet()
    styles['Title'].textColor = colors.HexColor('#12354a')
    styles['BodyText'].leading = 15
    story = []
    def p(text, style='BodyText'):
        story.extend([Paragraph(escape(text), styles[style]), Spacer(1, 10)])
    def table(rows):
        t = Table(rows, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#12354a')),
                              ('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),9),
                              ('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),9),
                              ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf5f7'),colors.white])]))
        story.extend([t, Spacer(1, 14)])
    p('AI-Based Industrial Equipment Failure Detection', 'Title')
    p('Reproducible research prototype | AI4I 2020 | Experiment seed 42', 'Heading2')
    p('1. Research problem', 'Heading2')
    p('Can six operating-condition features distinguish machine-failure records from normal records in simulated industrial data? This is snapshot classification, not a forecast with a defined lead time. No real machine or sensor was used.')
    p('2. Dataset and leakage prevention', 'Heading2')
    p(f'UCI AI4I 2020 contains {m["rows"]:,} synthetic records, with {m["failures"]} failure labels ({m["failures"]/m["rows"]:.2%}). Inputs are machine type, air and process temperature, rotational speed, torque and tool wear. UDI and Product ID are excluded. TWF, HDF, PWF, OSF and RNF never enter the model; they are used only for post-test error analysis. The original labels are retained unchanged.')
    p('3. Experimental design', 'Heading2')
    p('A fixed, stratified 60/20/20 split creates 6,000 training, 2,000 validation and 2,000 test records. One-hot encoding and scaling are fitted only on training data. Candidates are a prior baseline, balanced logistic regression, a depth-limited decision tree and a 350-tree random forest. For each learned model, the validation threshold maximizes F2 over 0.05-0.95 in 0.01 increments. F2 emphasizes recall; it is not a quantified maintenance cost. The winning model is chosen by validation F2 before test evaluation. There is no refit on validation or test data.')
    p(f'Selected model: {m["selected_model"]}; decision threshold: {m["threshold"]:.2f}. Random splits can overestimate generalization because neighboring synthetic records share operating regimes. Future work must include chronological and machine-held-out evaluation on real data.')
    story.append(PageBreak())
    p('4. Held-out results', 'Heading1')
    rows = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AP']]
    for name, values in m['test'].items():
        rows.append([name] + [f'{values[k]:.3f}' for k in ['accuracy','precision','recall','f1','average_precision']])
    table(rows)
    p('All results use each model\'s validation-selected threshold (baseline: 0.50). AP means average precision. Accuracy alone is misleading because normal records dominate the dataset.')
    story.append(Image(str(ROOT / 'reports/confusion_matrix.png'), width=4.8*inch, height=3.6*inch))
    p(f'The selected model detects {tp} of {tp+fn} failures, misses {fn}, and raises {fp} false alarms among {tn+fp} normal records. Recall bootstrap 95% interval: {m["recall_bootstrap_95_ci"][0]:.3f}-{m["recall_bootstrap_95_ci"][1]:.3f}. This interval resamples test positives only; it excludes training and dataset uncertainty.')
    story.append(PageBreak())
    p('5. False-negative analysis', 'Heading1')
    p(f'{fn} failed records received a score below the decision threshold. Their original inputs and model scores are saved in reports/false_negatives.csv. Failure-mode counts below can overlap, and some labels may have no listed mode. Indicators are never used as predictors.')
    table([['Failure mode', 'Missed records']] + [[k, str(v)] for k,v in m['false_negative_modes'].items()])
    p('Random failures and stochastic tool-wear outcomes are not fully inferable from these six features. Inspect low-score misses separately from near-threshold misses before proposing feature changes; any revised model needs a new untouched evaluation set.')
    story.append(Image(str(ROOT / 'reports/precision_recall.png'), width=6.3*inch, height=4.05*inch))
    story.append(PageBreak())
    p('6. Interpretation and implementation', 'Heading1')
    story.append(Image(str(ROOT / 'reports/feature_importance.png'), width=6.3*inch, height=3.54*inch))
    p('Permutation importance measures test average-precision loss after shuffling a feature. Correlated temperature features can share importance. This is a global association, not a causal explanation or a per-machine diagnosis. Streamlit and FastAPI share one validated predictor and SQLite history store. Scores are uncalibrated model outputs and should not be read as real-world failure probabilities.')
    p('7. Limitations and conclusion', 'Heading2')
    p('This prototype demonstrates a complete, reproducible ML application on a small synthetic dataset. It does not validate safety, estimate remaining useful life, guarantee uptime, or establish generalization to a real factory. No real-world cost model or probability calibration was tested. The local API has no authentication; keep it on localhost. Production use requires representative labeled sensor histories, temporal validation, calibrated uncertainty, monitoring, access control and domain review.')
    p('8. Reproducibility and references', 'Heading2')
    p(f'Python {m["python_version"]}; scikit-learn {m["sklearn_version"]}; model version {m["model_version"]}. Dataset SHA-256: {m["dataset_sha256"]}. See requirements.lock.txt, train.py, metrics.json and split_assignments.csv.')
    p('AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository. DOI: 10.24432/C5HS5C. https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset. License: Creative Commons Attribution 4.0 International. Matzka, S. (2020), Explainable Artificial Intelligence for Predictive Maintenance Applications, International Conference on Artificial Intelligence for Industries.')
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#536777'))
        canvas.drawString(42, 28, 'Industrial failure detection | Synthetic-data prototype')
        canvas.drawRightString(570, 28, str(doc.page))
    SimpleDocTemplate(str(ROOT / 'research_report.pdf'), rightMargin=42, leftMargin=42,
                      topMargin=36, bottomMargin=48).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    build_report()
