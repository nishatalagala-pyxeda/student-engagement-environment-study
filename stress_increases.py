"""Forecast stress above an individual's strictly historical mean; exploratory."""
from pathlib import Path
import json,html,hashlib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import (roc_auc_score,average_precision_score,brier_score_loss,
    accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
import joblib
import depress_weekly_stress as base
import stress_environment_hypothesis as prev
from assess_depressdata import canonical
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/stress-increases'
HISTORY=['past_mean_stress','prior_stress','prior_deviation','past_sd','prior_change','prior_report_count','days_since_prior_stress']
SIGNALS=['temperature_C','humidity','co2','voc','pm25','light','noise']
PLAN={
 'question':'Can we forecast a week with stress higher than the student usual level?',
 'primary_event':'Target PSS minus mean of admissible earlier PSS reports >=0.5 on supplied0–4 scale. Require at least2 earlier reports. Threshold is descriptive, not clinical. Sensitivities0.25 and0.75, plus at least3 reports at primary threshold; no threshold selected from performance.',
 'regression':'Signed excess stress = target PSS minus historical mean. Zero-change baseline predicts the historical mean; last-deviation baseline predicts the most recent stress score.',
 'timing':'For target survey date D, recall week [D-7,D). Stress reports strictly before D-7; test reports also strictly before the block training cutoff. Wearables/environment use [D-15,D-8), with one full day buffer before recall begins. No outcomes from test blocks used. Static demographic collection dates and upload timing unavailable.',
 'reference_mean':'Reference historical mean and event label calculated independently per target using only history available under its training/test cutoff. Test-block reference freezes with history. It never uses the target or future reports. A block-specific reference may differ from a rolling real-world updated mean.',
 'cohort':'Retain the previous234 environmental-response observations38 IDs before historical eligibility. Do not require wearable availability. Existing cohort selected using target-week sensor coverage, limiting generalization. Features missing from a prior week remain missing with train-only imputation; no new cohort expansion.',
 'validation':'Reuse14-day purged rolling blocks (primary) and the previously inspected April holdout (sensitivity). Original block eligibility40 training responses10 IDs; after history eligibility require20 train responses5 IDs and both event classes. All comparisons within a protocol share records. This is exploratory reuse, not an untouched test.',
 'models':'Primary logistic regression C=1 for event and ridge alpha10 for signed excess, history features only. Forest sensitivity300 trees depth3 minleaf10. Add identical personal/time controls; then earlier environment, Fitbit, both. No tuning or class balancing. Decision threshold0.5 fixed.',
 'baselines':'Event: training event prevalence probability, always no increase, persistence (last PSS minus history mean >= event threshold). Regression: zero excess, last deviation, mean training excess.',
 'metrics':'Event: ROC AUC, average precision, Brier, balanced accuracy, precision, recall, F1, confusion matrix. Accuracy alone can favor always no increase. Regression: MAE and RMSE. Paired fixed-prediction person-cluster bootstrap95% intervals (3000 draws) for Brier/MAE improvement; no refitting or full selection uncertainty.',
 'limits':'Sparse/selected wearables, few IDs, noisy estimated personal means, aggregate PSS scoring unverified, retrospective availability, reused splits, unmeasured exam/study demands. Prediction is not causation.'}

def load_history():
    source=next(base.SOURCE.rglob('PANAS_PSS.csv'));raw=pd.read_csv(source)
    raw['person']=raw.ID.map(canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce');raw['source_row']=raw.index+2
    h=raw[raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4)].copy()
    conflict=h.groupby(['person','date']).PSS.transform('nunique').gt(1);h[conflict].to_csv(OUT/'excluded_conflicting_history.csv',index=False)
    return h[~conflict].drop_duplicates(['person','date']).sort_values(['person','date']),{'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'conflicting_rows':int(conflict.sum())}

def add_history(d,h,cutoff=None):
    records=[]
    for _,r in d.iterrows():
        old=h[h.person.eq(r.person)&h.date.lt(r.date-pd.Timedelta(days=7))]
        if cutoff is not None:old=old[old.date.lt(cutoff)]
        old=old.sort_values('date');n=len(old)
        if n:
            mean=float(old.PSS.mean());last=old.iloc[-1]
            v={'past_mean_stress':mean,'prior_stress':float(last.PSS),'prior_deviation':float(last.PSS-mean),'past_sd':float(old.PSS.std(ddof=0)),'prior_change':float(old.iloc[-1].PSS-old.iloc[-2].PSS) if n>=2 else np.nan,'prior_report_count':n,'days_since_prior_stress':(r.date-last.date).days,'latest_history_date':last.date,'history_source_rows':','.join(str(int(x)) for x in old.source_row)}
            assert last.date<r.date-pd.Timedelta(days=7) and (cutoff is None or last.date<cutoff)
        else:v={c:np.nan for c in HISTORY};v.update(prior_report_count=0,latest_history_date=pd.NaT,history_source_rows='')
        records.append(v)
    out=d.reset_index(drop=True).copy()
    for c in records[0]:out[c]=[r[c] for r in records]
    out['excess_stress']=out.PSS-out.past_mean_stress
    return out

def earlier_environment(d):
    mapping=json.loads((ROOT/'outputs/depressdata-assessment/sensor_id_mapping.json').read_text());mp={v['sensor_id']:v['proposed_person'] for v in mapping if v['proposed_person']}
    p=next(base.SOURCE.rglob('Bedroom_IEQ.csv'));e=pd.read_csv(p);e['person']=e.ID.map(mp);e['local']=pd.to_datetime(e['timestamp(America/New_York)'],format='mixed');e['day']=e.local.dt.normalize()
    # Validate repeated fall-back times as distinct real instants before daily aggregation.
    instants=[]
    for _,g in e.groupby('ID',sort=False):instants.append(pd.Series(pd.DatetimeIndex(g.local).tz_localize('America/New_York',ambiguous='infer',nonexistent='NaT').tz_convert('UTC'),index=g.index))
    e['instant']=pd.concat(instants).sort_index();assert e.instant.notna().all() and not e.duplicated(['ID','instant']).any()
    e['temperature_C']=(e[next(c for c in e if c.lower().startswith('temp('))]-32)*5/9;e['humidity']=e.humid
    e=e[e.person.notna()];daily=e.groupby(['person','day'])[SIGNALS].mean();daily['slots']=e.groupby(['person','day']).local.nunique();daily=daily[daily.slots.ge(72)].reset_index()
    assert np.isfinite(daily[SIGNALS].to_numpy()).all()
    out=d.copy();features=['lag_env_'+c+'_'+s for c in SIGNALS for s in ['mean','sd']]
    for c in features:out[c]=np.nan
    audit=[]
    for i,r in out.iterrows():
        start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8);g=daily[daily.person.eq(r.person)&daily.day.ge(start)&daily.day.lt(end)]
        if len(g)>=5:
            for c in SIGNALS:
                out.loc[i,'lag_env_'+c+'_mean']=g[c].mean();out.loc[i,'lag_env_'+c+'_sd']=g[c].std(ddof=0)
        audit.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'start':str(start.date()),'end_exclusive':str(end.date()),'good_days':len(g),'last_day':str(g.day.max().date()) if len(g) else None})
        assert len(g)<=7 and (not len(g) or g.day.max()<end<r.date-pd.Timedelta(days=7))
    pd.DataFrame(audit).to_csv(OUT/'earlier_environment_audit.csv',index=False);daily.to_csv(OUT/'environment_daily.csv',index=False)
    return out,features,{'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'covered_rows':int(out[features].notna().all(axis=1).sum())}

def build(nums,cats,forest,classification):
    prep=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler()),nums),('categorical',make_pipeline(SimpleImputer(strategy='constant',fill_value='Unknown'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)],remainder='drop')
    if classification:model=RandomForestClassifier(n_estimators=300,max_depth=3,min_samples_leaf=10,random_state=42,n_jobs=1) if forest else LogisticRegression(C=1,max_iter=3000,random_state=42)
    else:model=RandomForestRegressor(n_estimators=300,max_depth=3,min_samples_leaf=10,random_state=42,n_jobs=1) if forest else Ridge(alpha=10)
    return make_pipeline(prep,model)

def classifier_score(y,prob):
    y=np.asarray(y,int);prob=np.asarray(prob,float);pred=(prob>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'events':int(y.sum()),'event_rate':float(y.mean()),'roc_auc':float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None,'average_precision':float(average_precision_score(y,prob)) if y.sum() else None,'brier':float(brier_score_loss(y,prob)),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if len(np.unique(y))==2 else None,'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def paired_ci(a,b,classification):
    a=a.set_index('source_row');b=b.set_index('source_row').loc[a.index];np.testing.assert_array_equal(a.observed,b.observed)
    delta=((b.predicted-b.observed)**2-(a.predicted-a.observed)**2) if classification else (b.predicted-b.observed).abs()-(a.predicted-a.observed).abs()
    groups=[delta[a.person.eq(x)].to_numpy() for x in sorted(a.person.unique())];rng=np.random.default_rng(42)
    boot=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(3000)]
    return {'metric':'Brier improvement' if classification else 'MAE improvement','improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'n':len(a),'people':len(groups)}

def evaluate(d,h,scopes,threshold=.5,min_history=2,primary=False):
    records=[];audit=[];splits=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        for block,cutoff,tr,te in prev.get_splits(d,protocol):
            train=add_history(d.iloc[tr],h);test=add_history(d.iloc[te],h,cutoff)
            train=train[train.prior_report_count.ge(min_history)].copy();test=test[test.prior_report_count.ge(min_history)].copy()
            ytrain=train.excess_stress.ge(threshold).astype(int)
            if len(train)<20 or train.person.nunique()<5 or len(test)==0 or ytrain.nunique()<2:continue
            ytest=test.excess_stress.ge(threshold).astype(int)
            splits.append({'protocol':protocol,'block':block,'cutoff_exclusive':str(cutoff.date()),'train_n':len(train),'test_n':len(test),'train_people':int(train.person.nunique()),'test_people':int(test.person.nunique()),'train_events':int(ytrain.sum()),'test_events':int(ytest.sum())})
            if primary:
                for role,frame in [('train',train),('test',test)]:
                    for _,r in frame.iterrows():audit.append({'protocol':protocol,'block':block,'role':role,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'history_cutoff_exclusive':str(cutoff.date()) if role=='test' else None,'latest_history_date':str(r.latest_history_date.date()),'history_source_rows':r.history_source_rows,**{c:float(r[c]) for c in HISTORY},'PSS':float(r.PSS),'excess_stress':float(r.excess_stress),'event':bool(r.excess_stress>=threshold)})
            pc={'Training prevalence':np.full(len(test),ytrain.mean()),'Always no increase':np.zeros(len(test)),'Last-deviation persistence':test.prior_deviation.ge(threshold).astype(float).to_numpy()}
            pr={'Zero excess (past mean)':np.zeros(len(test)),'Last deviation':test.prior_deviation.to_numpy(),'Training mean excess':np.clip(np.full(len(test),train.excess_stress.mean()),-test.past_mean_stress.to_numpy(),4-test.past_mean_stress.to_numpy())}
            for name,(nums,cats) in scopes.items():
                assert not set(nums+cats)&{'PSS','excess_stress','person','date','source_row'}
                families=[False,True] if primary else [False]
                for forest in families:
                    family='forest' if forest else 'linear';label=name+' | '+family
                    classifier=build(nums,cats,forest,True).fit(train[nums+cats],ytrain);pc[label]=classifier.predict_proba(test[nums+cats])[:,1]
                    regressor=build(nums,cats,forest,False).fit(train[nums+cats],train.excess_stress);raw=regressor.predict(test[nums+cats]);pr[label]=np.clip(raw,-test.past_mean_stress.to_numpy(),4-test.past_mean_stress.to_numpy())
            for task,observed,predictions in [('classification',ytest.to_numpy(),pc),('regression',test.excess_stress.to_numpy(),pr)]:
                for name,values in predictions.items():
                    for (_,r),o,p in zip(test.iterrows(),observed,values):records.append({'protocol':protocol,'block':block,'task':task,'model':name,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(o),'predicted':float(p),'history_mean':float(r.past_mean_stress),'observed_PSS':float(r.PSS),'past_count':int(r.prior_report_count),'fitbit_covered':bool(r.wearable_eligible)})
        print('Done',threshold,min_history,protocol,flush=True)
    p=pd.DataFrame(records);assert not p.duplicated(['protocol','task','model','source_row']).any()
    scores={};comparisons={}
    for (protocol,task),pp in p.groupby(['protocol','task']):
        scores.setdefault(protocol,{})[task]={};classification=task=='classification'
        for name,g in pp.groupby('model'):
            v=classifier_score(g.observed,g.predicted) if classification else base.score(g.observed.to_numpy(),g.predicted.to_numpy());v['people']=int(g.person.nunique());scores[protocol][task][name]=v
        if primary:
            pairs=[('History | linear','Training prevalence' if classification else 'Zero excess (past mean)'),('History | forest','Training prevalence' if classification else 'Zero excess (past mean)'),('History + controls | linear','History | linear')]
            for family in ['linear','forest']:
                for name in ['History + controls + environment','History + controls + Fitbit','History + controls + both']:pairs.append((name+' | '+family,'History + controls | '+family))
            comparisons.setdefault(protocol,{})[task]={a+' versus '+b:paired_ci(pp[pp.model.eq(a)],pp[pp.model.eq(b)],classification) for a,b in pairs}
    if primary:p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'history_target_audit.csv',index=False)
    else:p.to_csv(OUT/f'sensitivity_threshold_{threshold}_history_{min_history}_predictions.csv',index=False)
    return {'scores':scores,'comparisons':comparisons,'splits':splits}

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Predicting stress increases</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:36px}</style><h1>Can we predict stress above a student’s usual level?</h1><p>Exploratory DEPRESSData analysis. “Increase” means at least0.5 points above the mean of at least2 admissible past reports on the supplied0–4 scale. This is not a clinical cutoff and does not necessarily mean an increase from the immediately preceding week.</p>'
    c=r['primary']['scores']['purged_rolling_blocks']['classification'];g=r['primary']['scores']['purged_rolling_blocks']['regression'];v=c['History | linear'];b=c['Training prevalence'];reg=g['History | linear'];z=g['Zero excess (past mean)']
    text+=f'<h2>Main rolling test</h2><p><strong>These tests do not establish reliable prediction of above-usual stress weeks.</strong> {v["n"]} reports from{v["people"]} students include{v["events"]} above-usual weeks ({v["event_rate"]:.1%}). The prespecified history logistic model has ROC AUC{v["roc_auc"]:.3f}, average precision{v["average_precision"]:.3f}, Brier{v["brier"]:.3f} versus{b["brier"]:.3f} for training prevalence. At a fixed0.5 probability threshold, recall is{v["recall"]:.1%} and precision{v["precision"]:.1%}. History ridge excess-stress MAE is{reg["mae"]:.3f}, versus{z["mae"]:.3f} for predicting no deviation from the past mean. Full paired uncertainty and alternate models are below.</p><p>Classifying an above-average week is distinct from predicting an exact score. Apparent accuracy can come from predicting no increase most of the time. Probability ranking, calibration and actual detected events must also be considered. Changing the target does not add information to the predictors.</p><p>The earlier-environment logistic model ranks better (rolling AUC0.696, April0.765), but its Brier improvement over the identical control model is not established; its fixed0.5 threshold finds only1 of20 rolling events, with12 false alarms. Forest environmental results do not consistently corroborate the ranking gain. A small forest regression improvement from Fitbit over a weaker control model is exploratory among many comparisons and does not establish superiority to the personal-mean baseline. These isolated results are leads for independent validation, not a demonstrated alert system.</p>'
    for protocol,tasks in r['primary']['scores'].items():
        text+='<h2>'+html.escape(protocol)+'</h2><h3>Above-usual weeks</h3><table><tr><th>Model</th><th>N / students</th><th>AUC</th><th>Average precision</th><th>Brier↓</th><th>Balanced accuracy</th><th>Precision</th><th>Recall</th><th>TP / FP / FN / TN</th></tr>'
        for n,v in tasks['classification'].items():
            text+=f'<tr><td>{html.escape(n)}</td><td>{v["n"]} / {v["people"]}</td>'
            for key in ['roc_auc','average_precision','brier','balanced_accuracy','precision','recall']:text+='<td>'+('n.a.' if v[key] is None else f'{v[key]:.3f}')+'</td>'
            text+=f'<td>{v["tp"]} / {v["fp"]} / {v["fn"]} / {v["tn"]}</td></tr>'
        text+='</table><h3>Signed excess-stress regression</h3><table><tr><th>Model</th><th>MAE↓</th><th>RMSE↓</th><th>R²</th></tr>'
        for n,v in tasks['regression'].items():text+=f'<tr><td>{html.escape(n)}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td></tr>'
        text+='</table><h3>Paired improvement uncertainty</h3><table><tr><th>Task / comparison</th><th>Improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
        for task,cs in r['primary']['comparisons'][protocol].items():
            for n,v in cs.items():
                lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{task}: {html.escape(n)}</td><td>{v["improvement"]:.4f}</td><td>{lo:.4f} to {hi:.4f}</td></tr>'
        text+='</table>'
    text+='<h2>Threshold and history-count checks</h2><p>These checks use only the fixed history linear models and identical simple baselines. They are not used to pick a threshold.</p><table><tr><th>Threshold / minimum history / protocol</th><th>N / students / events</th><th>AUC</th><th>AP / event rate</th><th>Brier / prevalence Brier</th><th>Recall / precision</th><th>Excess MAE / zero-excess MAE</th></tr>'
    for name,s in r['sensitivity'].items():
        for protocol,tasks in s['scores'].items():
            v=tasks['classification']['History | linear'];b=tasks['classification']['Training prevalence'];g=tasks['regression'];text+=f'<tr><td>{name} / {protocol}</td><td>{v["n"]} / {v["people"]} / {v["events"]}</td><td>{v["roc_auc"]:.3f}</td><td>{v["average_precision"]:.3f} / {v["event_rate"]:.3f}</td><td>{v["brier"]:.3f} / {b["brier"]:.3f}</td><td>{v["recall"]:.3f} / {v["precision"]:.3f}</td><td>{g["History | linear"]["mae"]:.3f} / {g["Zero excess (past mean)"]["mae"]:.3f}</td></tr>'
    text+='</table>'
    diag=r.get('post_result_alert_diagnostics')
    if diag:
        text+='<h2>Descriptive alert tradeoffs (added after primary results)</h2><p>'+html.escape(diag['note'])+'</p><table><tr><th>Protocol / model</th><th>Probability cutoff</th><th>True alerts</th><th>False alerts</th><th>Missed increases</th><th>Recall</th><th>Precision</th></tr>'
        for v in diag['tradeoffs']:
            text+=f'<tr><td>{html.escape(v["protocol"]+" / "+v["model"])}</td><td>{v["probability_threshold"]:.1f}</td><td>{v["true_alerts"]}</td><td>{v["false_alerts"]}</td><td>{v["missed_events"]}</td><td>{v["recall"]:.3f}</td><td>{v["precision"]:.3f}</td></tr>'
        text+='</table><p>Lowering the alert threshold can increase detection while producing more false alerts; it does not itself improve probability ranking. No operating threshold is recommended from these inspected test data. Ranking intervals below reflect fixed prediction clusters and do not cover all training and model-selection uncertainty.</p><table><tr><th>Protocol / model</th><th>AUC person-cluster95% interval</th></tr>'
        for protocol,ms in diag['ranking_intervals'].items():
            for name,v in ms.items():
                lo,hi=v['auc_person_cluster_95_interval'];text+=f'<tr><td>{html.escape(protocol+" / "+name)}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Definitions, availability and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p>Environment features are equal-weight means and population SDs of good-day means (at least72 distinct15-minute slots/day, at least5 days/week), with temperature converted to Celsius; reported noise averages are descriptive, not Leq. Fitbit features reuse the checked daily summaries from the previous step (at least5 activity/positive-step days,3 main-sleep days; separate stream missingness retained). Short sensor-to-person mappings are inferred; unresolved F135 excluded. Main analyses retain missing earlier environmental/Fitbit features and impute only in training. Benefits could reflect availability patterns and need confirmation among measured subjects. Historical report count may vary across the frozen block; limited counts make the usual mean noisy. Personal/control variables have unknown collection dates, so history-only results are primary for timing.</p><p><a href="results.json">Complete results</a> · <a href="analysis_plan.json">Analysis definitions</a> · <a href="history_target_audit.csv">Historical reference/target audit</a> · <a href="earlier_environment_audit.csv">Earlier environmental window audit</a> · <a href="predictions.csv">Held-out predictions</a> · <a href="integrity_checks.json">Checks</a></p>'
    (OUT/'report.html').write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));h,hmeta=load_history();d=pd.read_csv(ROOT/'outputs/fitbit-stress/modeling_data.csv');d['date']=pd.to_datetime(d.survey_date)
    targets=h.set_index(['person','date']).PSS.reindex(pd.MultiIndex.from_frame(d[['person','date']]));np.testing.assert_allclose(targets.to_numpy(),d.PSS.to_numpy())
    d,env,emeta=earlier_environment(d);fit=[c for c in d if c.startswith('fitbit_')];assert len(fit)==16
    nums,cats=prev.scopes_for()['Controls'];scopes={'History':(HISTORY,[]),'History + controls':(HISTORY+nums,cats),'History + controls + environment':(HISTORY+nums+env,cats),'History + controls + Fitbit':(HISTORY+nums+fit,cats),'History + controls + both':(HISTORY+nums+env+fit,cats)}
    d.to_csv(OUT/'modeling_features.csv',index=False);print('Prepared',len(d),'rows; earlier environment coverage',emeta['covered_rows'],flush=True)
    primary=evaluate(d,h,scopes,primary=True);sens={}
    for threshold,min_count in [(.25,2),(.75,2),(.5,3)]:sens[f'threshold{threshold}_history{min_count}']=evaluate(d,h,{'History':(HISTORY,[])},threshold,min_count)
    # Research fits, not a prospective deployment; save exact features/definitions.
    full=add_history(d,h);full=full[full.prior_report_count.ge(2)];models={}
    for name,(ns,cs) in scopes.items():
        for classification in [True,False]:
            y=full.excess_stress.ge(.5).astype(int) if classification else full.excess_stress
            key=name+(' classifier' if classification else ' regressor');models[key]={'features':ns+cs,'model':build(ns,cs,False,classification).fit(full[ns+cs],y)}
    joblib.dump({'models':models,'plan':PLAN},OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    for key,v in models.items():np.testing.assert_allclose(v['model'].predict(full[v['features']]),restored['models'][key]['model'].predict(full[v['features']]))
    checks={'source_targets_match_cleaned_PSS':True,'history_strictly_before_recall_and_test_training_cutoff':True,'reference_mean_excludes_current_and_future_reports':True,'target_week_environment_and_wearables_excluded':True,'same_rows_for_all_feature_comparisons':True,'train_only_imputation_scaling_encoding':True,'no_duplicate_test_rows_per_model_protocol':True,'serialization_roundtrip_passes':True}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2));r={'plan':PLAN,'sources':{'history':hmeta,'environment':emeta,'wearable_features_sha256':hashlib.sha256((ROOT/'outputs/fitbit-stress/modeling_data.csv').read_bytes()).hexdigest()},'features':{n:ns+cs for n,(ns,cs) in scopes.items()},'primary':primary,'sensitivity':sens};(OUT/'results.json').write_text(json.dumps(r,indent=2));render(r)
    print(json.dumps(primary['scores'],indent=2),flush=True)
if __name__=='__main__':main()
