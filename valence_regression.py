"""Regression on the original PAM valence scores using verified broad environmental inputs."""
from pathlib import Path
import argparse, hashlib, html, json, os
import numpy as np
import pandas as pd
# All estimators use one worker; avoid initializing an unused process-pool runtime.
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import joblib
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from broad_environment_model import WEATHER_FEATURES, INDOOR_FEATURES, TIME_FEATURES

ROOT=Path(__file__).resolve().parent
PRIMARY='Environment ridge'

def specifications():
    def ridge():
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
    return {
        'Mean baseline':(TIME_FEATURES,DummyRegressor(strategy='mean')),
        'Median baseline':(TIME_FEATURES,DummyRegressor(strategy='median')),
        'Time of day ridge':(TIME_FEATURES,ridge()),
        'Outdoor ridge':(WEATHER_FEATURES,ridge()),
        'Campus indoor ridge':(INDOOR_FEATURES,ridge()),
        PRIMARY:(WEATHER_FEATURES+INDOOR_FEATURES,ridge()),
        'Environment + time ridge':(WEATHER_FEATURES+INDOOR_FEATURES+TIME_FEATURES,ridge()),
        'Environment forest':(WEATHER_FEATURES+INDOOR_FEATURES,make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=300,max_depth=3,min_samples_leaf=10,max_features=1.0,random_state=42,n_jobs=1))),
    }

def predict_valence(payload, frame):
    """Predict on prepared features, retaining the fixed known 1–4 scale bounds."""
    return np.clip(payload['model'].predict(frame[payload['features']]), *payload['bounds'])

def score(y,pred):
    return {'n':len(y),'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),'r2':float(r2_score(y,pred)),'bias':float(np.mean(pred-y))}

def evaluate(df, specs, protocol):
    if protocol=='held_out_date':
        splits=[(np.flatnonzero(df.date.ne(v)),np.flatnonzero(df.date.eq(v))) for v in sorted(df.date.unique())]
    elif protocol=='future_block':
        splits=[(np.flatnonzero(df.timestamp.lt('2019-11-02')),np.flatnonzero(df.timestamp.ge('2019-11-18')))]
    elif protocol=='held_out_student':
        splits=[(np.flatnonzero(df.Pid.ne(v)),np.flatnonzero(df.Pid.eq(v))) for v in sorted(df.Pid.unique())]
    else:raise ValueError(protocol)
    results={};records=[];y=df.Valence.to_numpy(float)
    for name,(cols,template) in specs.items():
        pred=np.full(len(df),np.nan);rawpred=pred.copy()
        for train,test in splits:
            assert len(train) and len(test) and set(train).isdisjoint(test)
            if protocol=='held_out_date':assert set(df.iloc[train].date).isdisjoint(df.iloc[test].date)
            if protocol=='held_out_student':assert set(df.iloc[train].Pid).isdisjoint(df.iloc[test].Pid)
            if protocol=='future_block':assert df.iloc[train].timestamp.max()<df.iloc[test].timestamp.min()
            model=clone(template).fit(df.iloc[train][cols],y[train])
            rawpred[test]=model.predict(df.iloc[test][cols]);pred[test]=np.clip(rawpred[test],1,4)
        used=np.isfinite(pred);results[name]=score(y[used],pred[used]);results[name]['predictions_clipped']=int(((rawpred[used]<1)|(rawpred[used]>4)).sum())
        for i in np.flatnonzero(used):
            records.append({'protocol':protocol,'model':name,'source_row':int(df.iloc[i].source_row),'participant':int(df.iloc[i].Pid),'date':df.iloc[i].date,'observed_valence':float(y[i]),'predicted_valence':float(pred[i]),'unbounded_prediction':float(rawpred[i])})
    return results,pd.DataFrame(records)

def interval(predictions,baseline):
    p=predictions[predictions.protocol.eq('held_out_date')]
    a=p[p.model.eq(PRIMARY)].set_index('source_row');b=p[p.model.eq(baseline)].set_index('source_row').loc[a.index]
    d=pd.DataFrame({'date':a.date,'delta':np.abs(a.predicted_valence-a.observed_valence)-np.abs(b.predicted_valence-b.observed_valence)})
    values=[d[d.date.eq(v)].delta.to_numpy() for v in sorted(d.date.unique())];rng=np.random.default_rng(42)
    boot=[np.concatenate([values[i] for i in rng.integers(0,len(values),len(values))]).mean() for _ in range(3000)]
    return {'mae_difference':float(d.delta.mean()),'date_cluster_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'interpretation':'Negative favors regression. Resamples nine date clusters of fixed held-out predictions, without retraining.'}

def make_report(out,results,predictions,meta,ci,sensitivity):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(results['held_out_date']);fig,axes=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.3,1]})
    ypos=np.arange(len(names));axes[0].barh(ypos-.16,[results['held_out_date'][n]['mae'] for n in names],.32,color='#326a9a',label='Held-out date');axes[0].barh(ypos+.16,[results['future_block'][n]['mae'] for n in names],.32,color='#cc8a40',label='Later November block')
    axes[0].set(yticks=ypos,yticklabels=names,xlabel='Mean absolute error (lower is better)',title='Valence regression on the 1–4 scale');axes[0].invert_yaxis();axes[0].legend(fontsize=9)
    p=predictions[predictions.protocol.eq('held_out_date')&predictions.model.eq(PRIMARY)];rng=np.random.default_rng(42)
    axes[1].scatter(p.observed_valence+rng.uniform(-.06,.06,len(p)),p.predicted_valence,alpha=.55,s=26,color='#326a9a');axes[1].plot([1,4],[1,4],'--',color='#777',label='Perfect prediction');axes[1].set(xlim=(.8,4.2),ylim=(.8,4.2),xticks=[1,2,3,4],xlabel='Observed valence (display jitter only)',ylabel='Predicted valence',title='Primary model: held-out dates');axes[1].legend(fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=160);plt.close(fig)
    def table(protocol):
        rows=''.join('<tr><td>'+html.escape(n)+'</td>'+''.join(f'<td>{v[k]:.3f}</td>' for k in ['mae','rmse','r2'])+'</tr>' for n,v in results[protocol].items())
        return '<table><tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr>'+rows+'</table>'
    a=results['held_out_date'][PRIMARY];base=results['held_out_date']['Mean baseline'];future=results['future_block'][PRIMARY];low,high=ci['versus_mean']['date_cluster_bootstrap_95_interval']
    outcome='The primary environmental regression does not beat the mean-score baseline on held-out-date MAE.' if a['mae']>=base['mae'] else 'The primary environmental regression has lower observed held-out-date MAE than the mean-score baseline; the small date count limits conclusions.'
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Valence regression</title><style>body{{font:16px/1.6 system-ui;color:#253243;max-width:1150px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}img{{max-width:100%}}h2{{margin-top:30px}}</style></head><body>
    <h1>Predicting reported valence from broad environmental conditions</h1><p>Regression analysis, November 10, 2026.</p><p><strong>{outcome}</strong> Primary MAE is {a['mae']:.3f} versus {base['mae']:.3f} for the mean baseline; RMSE {a['rmse']:.3f}; R² {a['r2']:.3f}. On the chronological later-November test, primary MAE is {future['mae']:.3f}.</p>
    <h2>What changed from classification</h2><p>Predict the original PAM Valence score 1, 2, 3 or 4 as a numeric outcome, without binarizing it. Regression retains ordinal information but assumes that adjacent levels have approximately equal spacing. Scores are reported mood, not emotional engagement or a clinical scale. Predictions may be fractional (for example 2.7), and are clipped to the known 1–4 bounds before evaluation; clipping rules are fixed, not learned from test outcomes.</p>
    <p>The same 163 clean responses from 22 students on nine dates are used. Six conflicting same-student/same-timestamp responses remain excluded. A sensitivity check retaining all 169 labels is saved separately. Original labels and source files were not modified.</p>
    <h2>Inputs and evaluation</h2><p>Seven broad environmental inputs, identical to the classification study: outdoor temperature, humidity, wind speed, precipitation, solar radiation, and campus-median indoor temperature/humidity across the fixed eight-room panel. Each is a trailing six-reading average on the five-minute grid, ending at or before survey submission. Campus indoor variables are proxies, not measured student-room exposures. No participant IDs, survey answers, arousal labels, CO₂ or noise are predictor inputs.</p>
    <p>Primary model: standardized ridge regression, alpha=1. Comparator forest: 300 trees, depth 3, minimum leaf 10, seed 42. Include mean, median and time-of-day baselines, outdoor/indoor ablations and environment-plus-time ridge. Specifications were fixed before the regression results; no tuning or outcome-based feature selection. Preprocessing is fitted only on training folds.</p>
    <p>Main check leaves each whole date out. Chronological check trains on 105 October responses and tests on 58 responses from four later November dates. Students can occur on both sides of these date splits; these test different contexts for the same cohort. Supplemental leave-one-student-out results share dates and answer a different generalization question. MAE and RMSE are in valence-score units. Negative R² means predictions perform worse than the reference using the evaluated observations’ mean; it is not a negative correlation coefficient.</p>
    <h2>Held-out dates</h2>{table('held_out_date')}<h2>Chronological check</h2>{table('future_block')}<img src="evaluation.png" alt="MAE model comparison and observed versus predicted valence"><h2>Held-out students, supplemental</h2>{table('held_out_student')}
    <h2>Uncertainty and sensitivity</h2><p>Primary-minus-mean-baseline MAE difference: {ci['versus_mean']['mae_difference']:+.3f}, date-cluster bootstrap 95% interval [{low:+.3f}, {high:+.3f}]. Negative favors the regression. Only nine date clusters are resampled; models are not refitted, so this interval does not capture all training uncertainty. Results retaining ambiguous labels and unbounded-prediction counts are in metrics.json. The bound rule is also stored with the fitted model.</p>
    <h2>Limits</h2><p>Only nine dates, incomplete mood reporting and missing room mapping constrain the conclusions. Broad environment is shared by students and confounded with calendar/class activity. Regression cannot restore missing spatial exposure information or establish causality. These results are an exploratory test; no effective mood predictor is claimed solely from fitting the model. R² and errors from training on all rows are not used as evidence of generalization.</p>
    <h2>Outputs</h2><p><a href="metrics.json">Full metrics and provenance</a>; <a href="evaluation_predictions.csv">held-out predictions</a>; <a href="primary_model.joblib">fitted ridge model</a>. Run valence_regression.py to reproduce the analysis. For prepared feature data, use predict_valence(payload, frame) from that module, which selects the stored feature names and applies the fixed bounds. The previous classification model and report remain unchanged.</p></body></html>'''
    (out/'report.html').write_text(text)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=ROOT/'outputs/broad-environment-model/modeling_data.csv');parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/valence-regression');args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input,parse_dates=['timestamp','feature_end']);assert df.Valence.isin([1,2,3,4]).all();assert not df.duplicated(['Pid','timestamp']).any();assert df[WEATHER_FEATURES+INDOOR_FEATURES+TIME_FEATURES].notna().all().all();assert (df.feature_end<=df.timestamp).all()
    specs=specifications();results={};records=[]
    for protocol in ['held_out_date','future_block','held_out_student']:
        results[protocol],p=evaluate(df,specs,protocol);records.append(p);print('Completed',protocol,flush=True)
    predictions=pd.concat(records,ignore_index=True);ci={'versus_mean':interval(predictions,'Mean baseline'),'versus_median':interval(predictions,'Median baseline'),'versus_time':interval(predictions,'Time of day ridge')}
    # Reuse verified source preparation for the all-label sensitivity check.
    from broad_environment_model import make_dataset
    previous=json.loads((ROOT/'outputs/broad-environment-model/metrics.json').read_text())['metadata']
    regenerated,allrows=make_dataset(Path(previous['input_dir']),Path(previous['survey_zip']))
    cols=['source_row','Valence']+WEATHER_FEATURES+INDOOR_FEATURES+TIME_FEATURES
    np.testing.assert_allclose(regenerated[cols].to_numpy(float),df[cols].to_numpy(float),rtol=0,atol=1e-10)
    sensitivity={};sensitivity_n=len(allrows)
    assert sensitivity_n==169
    for protocol in ['held_out_date','future_block']:
        sensitivity[protocol],_=evaluate(allrows,{n:specs[n] for n in ['Mean baseline',PRIMARY]},protocol)
    meta={'target':'Original PAM Valence, numeric 1–4','n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'label_counts':df.Valence.value_counts().sort_index().to_dict(),'input_file':str(args.input),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'features':WEATHER_FEATURES+INDOOR_FEATURES,'bounds':[1,4],'sensitivity_n':sensitivity_n}
    fitted={n:clone(t).fit(df[cols],df.Valence) for n,(cols,t) in specs.items()};payload={'model':fitted[PRIMARY],'features':specs[PRIMARY][0],'bounds':(1,4),'metadata':meta}
    joblib.dump(payload,out/'primary_model.joblib');joblib.dump({'models':fitted,'features':{n:cols for n,(cols,_) in specs.items()},'bounds':(1,4),'metadata':meta},out/'comparison_models.joblib')
    restored=joblib.load(out/'primary_model.joblib');np.testing.assert_allclose(predict_valence(restored,df),predict_valence(payload,df));assert ((predict_valence(restored,df)>=1)&(predict_valence(restored,df)<=4)).all()
    predictions.to_csv(out/'evaluation_predictions.csv',index=False);(out/'metrics.json').write_text(json.dumps({'metadata':meta,'evaluation':results,'mae_difference_intervals':ci,'sensitivity_all_169':sensitivity},indent=2));make_report(out,results,predictions,meta,ci,sensitivity)
    print(json.dumps({'metadata':meta,'evaluation':results,'intervals':ci},indent=2),flush=True)

if __name__=='__main__':main()
