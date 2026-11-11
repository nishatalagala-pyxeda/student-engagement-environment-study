"""Reproducible exploratory valence modeling. Original data are never modified."""
from pathlib import Path
import argparse, hashlib, html, json, os, platform, zipfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
ROOMS = ['28', '29', '31', '43', 'KB2', 'KB3', 'KB4', 'KB5']
OUTDOOR = ['OutdoorTemperature', 'OutdoorHumidity', 'OutdoorWindSpeed', 'Precipitation', 'SolarRadiation']
INDOOR = ['IndoorTemperature', 'IndoorHumidity']
WEATHER_FEATURES = [c + '_trailing6_mean' for c in OUTDOOR]
INDOOR_FEATURES = ['CampusMedian' + c.removeprefix('Indoor') + '_trailing6_mean' for c in INDOOR]
TIME_FEATURES = ['clock_sin', 'clock_cos']
PRIMARY = 'Environment logistic'

def make_dataset(data_dir, survey_zip):
    frames = {}
    for room in ROOMS + ['19']:
        d = pd.read_csv(data_dir / 'longitudinal' / (room + '.csv'))
        dt = pd.Timestamp('2019-10-07') + pd.to_timedelta(d.Day - 1, unit='D') + pd.to_timedelta(d.Time)
        d.index = pd.DatetimeIndex(dt)
        assert d.index.is_unique and d.index.is_monotonic_increasing
        frames[room] = d
    env = frames['19'][OUTDOOR].copy()
    for room in ROOMS:
        assert frames[room].index.equals(env.index)
        np.testing.assert_allclose(frames[room][OUTDOOR], env[OUTDOOR], rtol=0, atol=1e-9)
    for c in INDOOR:
        matrix = pd.concat([frames[r][c].where(frames[r].UsabilityMask) for r in ROOMS], axis=1)
        # Fixed eight-room panel, all eight required: no changing room composition.
        env['CampusMedian' + c.removeprefix('Indoor')] = matrix.median(axis=1).where(matrix.notna().all(axis=1))
    features = env.rolling(6, min_periods=6).mean()
    features.columns = [c + '_trailing6_mean' for c in features]
    with zipfile.ZipFile(survey_zip) as z:
        raw = pd.read_csv(z.open('Survey/Student_survey.csv'))
    original = pd.read_csv(data_dir / 'survey/student_survey.csv')
    common = [c for c in raw if c != 'Time']
    assert raw[common].fillna('__NA__').equals(original[common].fillna('__NA__'))
    timestamps = pd.to_datetime(raw.Time, dayfirst=True)
    assert (pd.to_timedelta(timestamps.dt.strftime('%H:%M:%S')) == pd.to_timedelta(original.Time)).all()
    mood = raw.loc[raw.Valence.notna()].copy()
    mood['source_row'] = mood.index + 2
    mood['timestamp'] = timestamps.loc[mood.index]
    mood['date'] = mood.timestamp.dt.strftime('%Y-%m-%d')
    mood['feature_end'] = mood.timestamp.dt.floor('5min')
    # Only observations at/before submission are used, in the released rounded grid.
    assert (mood.feature_end <= mood.timestamp).all()
    joined = features.reindex(pd.DatetimeIndex(mood.feature_end))
    for c in features:
        mood[c] = joined[c].to_numpy()
    minutes = mood.timestamp.dt.hour * 60 + mood.timestamp.dt.minute
    mood['clock_sin'] = np.sin(2 * np.pi * minutes / 1440)
    mood['clock_cos'] = np.cos(2 * np.pi * minutes / 1440)
    assert mood.Valence.isin([1, 2, 3, 4]).all()
    mood['higher_valence'] = mood.Valence.ge(3).astype(int)
    groups = mood.groupby(['Pid', 'timestamp']).higher_valence.transform('nunique')
    mood['conflicting_label'] = groups > 1
    clean = mood.loc[~mood.conflicting_label].drop_duplicates(['Pid', 'timestamp']).copy()
    assert not clean.duplicated(['Pid', 'timestamp']).any()
    assert clean[WEATHER_FEATURES + INDOOR_FEATURES].notna().all().all(), 'Missing exposure: investigate instead of silently filling.'
    return clean.reset_index(drop=True), mood.reset_index(drop=True)

def specifications():
    def logistic():
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=42))
    return {
        'Majority baseline': (TIME_FEATURES, DummyClassifier(strategy='prior')),
        'Time of day logistic': (TIME_FEATURES, logistic()),
        'Outdoor logistic': (WEATHER_FEATURES, logistic()),
        'Campus indoor logistic': (INDOOR_FEATURES, logistic()),
        PRIMARY: (WEATHER_FEATURES + INDOOR_FEATURES, logistic()),
        'Environment + time logistic': (WEATHER_FEATURES + INDOOR_FEATURES + TIME_FEATURES, logistic()),
        'Environment forest': (WEATHER_FEATURES + INDOOR_FEATURES, make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=1))),
    }

def score(y, probability):
    pred = (probability >= .5).astype(int)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)), 'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None, 'brier': float(brier_score_loss(y, probability)), 'log_loss': float(log_loss(y, probability, labels=[0, 1])), 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}

def run_protocol(df, spec, protocol):
    y = df.higher_valence.to_numpy()
    if protocol == 'held_out_date':
        splits = [(np.flatnonzero(df.date.ne(day)), np.flatnonzero(df.date.eq(day))) for day in sorted(df.date.unique())]
    elif protocol == 'future_block':
        splits = [(np.flatnonzero(df.timestamp.lt('2019-11-01 23:59:59')), np.flatnonzero(df.timestamp.ge('2019-11-18')))]
    elif protocol == 'held_out_student':
        splits = [(np.flatnonzero(df.Pid.ne(pid)), np.flatnonzero(df.Pid.eq(pid))) for pid in sorted(df.Pid.unique())]
    else:
        raise ValueError(protocol)
    results, predictions = {}, []
    for name, (cols, template) in spec.items():
        probabilities = np.full(len(df), np.nan)
        for train, test in splits:
            assert len(train) and len(test) and not set(train).intersection(test)
            if protocol == 'held_out_date':
                assert set(df.iloc[train].date).isdisjoint(df.iloc[test].date)
            if protocol == 'future_block':
                assert df.iloc[train].timestamp.max() < df.iloc[test].timestamp.min()
            if protocol == 'held_out_student':
                assert set(df.iloc[train].Pid).isdisjoint(df.iloc[test].Pid)
            estimator = clone(template).fit(df.iloc[train][cols], y[train])
            probabilities[test] = estimator.predict_proba(df.iloc[test][cols])[:, list(estimator.classes_).index(1)]
        used = np.isfinite(probabilities)
        results[name] = score(y[used], probabilities[used])
        for i in np.flatnonzero(used):
            predictions.append({'protocol': protocol, 'model': name, 'source_row': int(df.iloc[i].source_row), 'date': df.iloc[i].date, 'participant': int(df.iloc[i].Pid), 'observed': int(y[i]), 'probability_higher': float(probabilities[i]), 'predicted': int(probabilities[i] >= .5)})
    return results, pd.DataFrame(predictions)

def clustered_accuracy_difference(predictions, model, baseline='Majority baseline'):
    p = predictions[predictions.protocol.eq('held_out_date')]
    a = p[p.model.eq(model)].set_index('source_row')
    b = p[p.model.eq(baseline)].set_index('source_row').loc[a.index]
    delta = pd.DataFrame({'date': a.date, 'delta': (a.predicted == a.observed).astype(int) - (b.predicted == b.observed).astype(int)})
    dates = sorted(delta.date.unique()); rng = np.random.default_rng(42)
    values = [delta[delta.date.eq(day)].delta.to_numpy() for day in dates]
    bootstrap = [np.concatenate([values[i] for i in rng.integers(0, len(values), len(values))]).mean() for _ in range(3000)]
    return {'difference': float(delta.delta.mean()), 'date_cluster_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(), 'note': 'Resamples nine date clusters of existing held-out predictions; does not refit models or capture full model-training uncertainty.'}

def report(outdir, metadata, results, sensitivity, intervals):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(results['held_out_date'])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(names)); h = .32
    ax.barh(y-h/2, [results['held_out_date'][n]['balanced_accuracy'] for n in names], h, label='Held-out date', color='#326a9a')
    ax.barh(y+h/2, [results['future_block'][n]['balanced_accuracy'] for n in names], h, label='Later November dates', color='#cc8a40')
    ax.set(yticks=y, yticklabels=names, xlim=(0, 1), xlabel='Balanced accuracy (0.50 = chance reference)', title='Broad environmental predictors of reported valence')
    ax.axvline(.5, color='#666', linestyle='--', linewidth=1); ax.invert_yaxis(); ax.legend(loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir/'evaluation.png', dpi=160); plt.close(fig)
    def table(protocol):
        rows=[]
        for name, v in results[protocol].items():
            rows.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[c]:.3f}</td>' for c in ['accuracy','balanced_accuracy','macro_f1','roc_auc','brier'])+'</tr>')
        return '<table><tr><th>Model</th><th>Accuracy</th><th>Balanced accuracy</th><th>Macro F1</th><th>ROC AUC</th><th>Brier (lower better)</th></tr>'+''.join(rows)+'</table>'
    primary=results['held_out_date'][PRIMARY]; baseline=results['held_out_date']['Majority baseline']; future=results['future_block'][PRIMARY]
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Broad environment mood models</title><style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#253243}}h1{{font-size:29px}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}</style></head><body>
    <h1>Broad environmental predictors of reported valence</h1><p>Exploratory analysis; trained and evaluated November 10, 2026.</p>
    <p><strong>Primary prespecified model:</strong> logistic regression using outdoor weather and campus-wide indoor proxies. Held-out-date accuracy is {primary['accuracy']:.1%}, versus {baseline['accuracy']:.1%} for the training-majority baseline; balanced accuracy is {primary['balanced_accuracy']:.1%}. On the later November block, primary accuracy is {future['accuracy']:.1%} and balanced accuracy is {future['balanced_accuracy']:.1%}. These are observed results, not established generalization guarantees.</p>
    <h2>Target and data</h2><p>Higher reported valence = PAM values 3–4; lower reported valence = 1–2. This is an exploratory threshold on the supplied ordinal mood scale, not a clinical diagnosis or emotional-engagement measure. Started with 169 mood-labeled surveys. Three participant/timestamp pairs have conflicting lower/higher labels: exclude all six responses rather than choose one. Main analysis: {metadata['n']} responses, {metadata['students']} students, {metadata['dates']} dates; {metadata['lower']} lower and {metadata['higher']} higher valence labels. Each response is one observation; no overlapping feature windows are split into separate training examples.</p>
    <h2>Inputs</h2><p>Outdoor temperature, humidity, wind speed, precipitation and solar radiation; campus median temperature and humidity across fixed rooms {', '.join(ROOMS)}. The indoor median requires all eight rooms to be usable at each timestamp. All seven inputs are six-reading trailing means (approximately 30 minutes), ending at the five-minute grid time at or before the survey. The release rounded original sensor timestamps to five-minute intervals, so timing is approximate. No CO₂, noise, wristband physiology, student ID, demographics, engagement answers, clothing or other mood labels enter the predictors. The campus indoor measures are context proxies, not each student's room conditions.</p>
    <p>Outdoor readings match across room files; room 19 supplies the canonical weather copy. Its room usability mask is not a quality flag for campus outdoor weather. Longitudinal dates are independently verified against the original Figshare archive. Survey rows/time-of-day are checked against the local release before using the original dates. No outcome-based feature selection was performed.</p>
    <h2>Evaluation design</h2><p>Main evaluation leaves one entire date out, repeated for all nine dates. Shared environmental contexts are therefore kept outside training for that date. Students can appear on other training dates: this tests new dates for the same cohort, not new students. A second, chronological check trains on October 28–November 1 ({metadata['chronological_train_n']} responses) and tests on November 18–22 ({metadata['chronological_test_n']} responses across four dates). All imputation, scaling and fitting use training data only. A supplemental leave-one-student-out check is reported separately; it shares dates across training and test and is not a substitute for the date-held-out analysis.</p>
    <p>Logistic regression C=1. Forest: 300 trees, maximum depth 3, minimum leaf size 10, random state 42. Parameters and input sets were fixed before evaluation; no hyperparameter search. The primary model remains environmental logistic regression regardless of which comparison scores highest. Probability threshold is 0.5. Metrics pool held-out predictions across responses; they do not treat repeated student/date observations as independent for uncertainty.</p>
    <h2>Held-out dates</h2>{table('held_out_date')}<h2>Chronological later-block test</h2>{table('future_block')}
    <img src="evaluation.png" alt="Balanced accuracy by model and evaluation design">
    <h2>Held-out students: supplemental</h2>{table('held_out_student')}
    <h2>Uncertainty and label sensitivity</h2><p>Primary-minus-majority accuracy difference: {intervals['vs_majority']['difference']:+.1%}; date-cluster bootstrap interval [{intervals['vs_majority']['date_cluster_bootstrap_95_interval'][0]:+.1%}, {intervals['vs_majority']['date_cluster_bootstrap_95_interval'][1]:+.1%}]. This resamples only nine dates of existing predictions, without refitting; uncertainty is substantial and training variability is not fully represented. See metrics.json for comparison with the time-only baseline and for sensitivity results retaining all 169 responses, including conflicting labels.</p>
    <h2>Interpretation limits</h2><ul><li>Only nine dates and 22 students supplied mood labels; the 163 responses do not equal 163 independent weather contexts. No room-specific indoor exposure can be inferred.</li><li>Weather, calendar periods, class schedules and other shared factors are confounded. Prediction does not establish that environmental conditions cause mood changes.</li><li>Label missingness is substantial and may be selective. PAM labels were available only in the later study weeks. The binary threshold discards ordinal detail.</li><li>Repeated students, shared timestamps, interpolated environmental readings and unverified physical sensor accuracy limit conclusions. No new-school or prospective validation exists.</li><li>The full-data fitted artifacts are for research and further validation. Their training-set performance is not reported as evidence of predictive success.</li></ul>
    <h2>Saved outputs and provenance</h2><p><a href="metrics.json">Metrics and metadata</a>; <a href="evaluation_predictions.csv">held-out predictions</a>; <a href="modeling_data.csv">prepared modeling observations</a>; <a href="primary_model.joblib">fitted primary model</a>. The model payload includes the required input names, target definition and provenance. Raw dataset files were not modified. Input paths and software versions are recorded in metrics.json. Run broad_environment_model.py from the project environment to reproduce outputs; README-model.md describes the workflow.</p></body></html>'''
    (outdir/'report.html').write_text(text)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',type=Path); parser.add_argument('--survey-zip',type=Path,default=Path('/Users/nishatalagala/Downloads/Figshare-Navya/Survey.zip')); parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/broad-environment-model'); args=parser.parse_args()
    data_dir=args.data_dir or next(p for p in Path('/Users/nishatalagala/Downloads').glob('in-gauge*') if p.is_dir())
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    df,all_labels=make_dataset(data_dir,args.survey_zip); spec=specifications(); results={};preds=[]
    for protocol in ['held_out_date','future_block','held_out_student']:
        results[protocol],pred=run_protocol(df,spec,protocol);preds.append(pred);print('Completed',protocol,flush=True)
    predictions=pd.concat(preds,ignore_index=True)
    sensitivity={}
    for protocol in ['held_out_date','future_block']:
        sensitivity[protocol],_=run_protocol(all_labels,{n:spec[n] for n in ['Majority baseline',PRIMARY]},protocol)
    metadata={'n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'lower':int(df.higher_valence.eq(0).sum()),'higher':int(df.higher_valence.eq(1).sum()),'excluded_conflicting_responses':int(all_labels.conflicting_label.sum()),'missing_feature_cells':int(df[WEATHER_FEATURES+INDOOR_FEATURES].isna().sum().sum()),'chronological_train_n':int(df.timestamp.lt('2019-11-01 23:59:59').sum()),'chronological_test_n':int(df.timestamp.ge('2019-11-18').sum()),'input_dir':str(data_dir),'survey_zip':str(args.survey_zip),'survey_zip_sha256':hashlib.sha256(args.survey_zip.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,'features':WEATHER_FEATURES+INDOOR_FEATURES,'date_counts':df.groupby('date').size().to_dict()}
    intervals={'vs_majority':clustered_accuracy_difference(predictions,PRIMARY),'vs_time_only':clustered_accuracy_difference(predictions,PRIMARY,'Time of day logistic')}
    fitted={}
    for name,(cols,template) in spec.items():
        fitted[name]=clone(template).fit(df[cols],df.higher_valence)
    payload={'model':fitted[PRIMARY],'features':spec[PRIMARY][0],'target':'higher PAM valence (3–4) versus lower (1–2)','metadata':metadata,'warning':'Campus context proxies; exploratory model with nine labeled dates. Inputs require the documented preparation.'}
    joblib.dump(payload,out/'primary_model.joblib');joblib.dump({'models':fitted,'feature_sets':{n:cols for n,(cols,_) in spec.items()},'metadata':metadata},out/'comparison_models.joblib')
    # Confirm serialized model round-trip and required input shape.
    restored=joblib.load(out/'primary_model.joblib');np.testing.assert_allclose(restored['model'].predict_proba(df[restored['features']]),fitted[PRIMARY].predict_proba(df[spec[PRIMARY][0]]))
    df.to_csv(out/'modeling_data.csv',index=False);predictions.to_csv(out/'evaluation_predictions.csv',index=False)
    (out/'excluded_conflicting_labels.csv').write_text(all_labels.loc[all_labels.conflicting_label,['source_row','Pid','timestamp','Valence','higher_valence']].to_csv(index=False))
    (out/'metrics.json').write_text(json.dumps({'metadata':metadata,'evaluation':results,'sensitivity_all_169':sensitivity,'accuracy_difference_intervals':intervals},indent=2))
    report(out,metadata,results,sensitivity,intervals)
    print(json.dumps({'metadata':metadata,'evaluation':results,'intervals':intervals},indent=2),flush=True)

if __name__=='__main__':main()
