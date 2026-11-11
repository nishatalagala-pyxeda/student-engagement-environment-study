"""Predict above-usual stress using strictly earlier personal environmental changes."""
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
import joblib
import expanded_stress_increases as core
old=core.old;ROOT=core.ROOT;OUT=ROOT/'outputs/personal-environment-changes'
SIGNALS=old.SIGNALS
PLAN={
 'question':'Do departures from a student usual room conditions improve above-usual stress prediction?',
 'target':'Unchanged: PSS >=0.5 above historical mean, at least2 admissible earlier stress reports. Also regress signed excess over that mean. Descriptive, not clinical; not necessarily a rise from the immediately preceding report.',
 'recent_environment':'For survey D use [D-15,D-8), with at least5 good days; good day >=72 distinct15-minute sensor slots. Same window as earlier environmental tests, entirely before recall week [D-7,D).',
 'usual_environment':'For each target independently, usual conditions use the28 calendar days strictly before the recent environmental window: [D-43,D-15). Require at least14 good days. No current-window or later readings enter this reference. Reference is a recent personal baseline, not the full-study average. This threshold is a fixed analysis choice, not a validated wear/exposure criterion.',
 'new_features':'21 features: for7 signals, recent mean minus prior28-day mean; recent daily-mean population SD minus prior28-day daily-mean population SD; recent daily-mean linear trend per7 calendar days using actual elapsed-day gaps. Computed only when both windows qualify. Raw reported units, temperature Celsius; arithmetic noise summaries not Leq. SD contrasts over unequal-duration windows are descriptive.',
 'scopes':'History only; History+previous14 environmental mean/SD features; History+21 changes; History+previous14 environmental features+21 changes. Same fixed logistic(C1)/ridge(alpha10) primary, shallow forest(300trees depth3 leaf10 seed42) sensitivity, no tuning or class balancing. Probability threshold0.5 unchanged.',
 'training':'All earlier cleaned stress reports irrespective of sensors, same expanded training pools. Missing environmental features remain missing with training-only median imputation and missingness indicators. Student IDs, reference dates/counts and current/future stress not model inputs. Prior14-feature/history predictions reused from exact earlier saved expanded models.',
 'validation':'Same14-day expanding calendar blocks anchoredJune1,2020,7-day training gap and minimum training counts. Same reused April test. Stress history strictly before D-7 and frozen before block training cutoff. Sensor features/reference may update from earlier observed readings within a test block, as in prior environmental tests; no stress labels update.',
 'primary_comparison':'On identical test rows with both current and personal-reference sensor coverage, compare adding changes to History+previous14environment features. Also compare replacement changes versus History alone and versus previous environmental features. All-test-row results are secondary because missingness can dominate.',
 'metrics':'ROC AUC, AP, Brier, precision/recall/confusion at0.5; excess MAE/RMSE. Paired fixed-prediction person-cluster bootstrap3000draws seed42 for Brier/MAE gain. Positive gain favors changes. No refitting or full model-selection uncertainty.',
 'limitations':'Selected small sensor cohort, inferred sensor IDs, unknown actual room occupancy, noisy estimated personal baselines, time-varying confounding and repeated exploratory analysis. Not a causal test. Different coverage subsets cannot be compared as if model improvements. Future readings and full-study person centering are prohibited.'}

def prepare():
    d=pd.read_csv(core.OUT/'environment_modeling_features.csv',float_precision='round_trip');d.date=pd.to_datetime(d.date)
    p=ROOT/'outputs/stress-increases/environment_daily.csv';daily=pd.read_csv(p,float_precision='round_trip');daily.day=pd.to_datetime(daily.day)
    names=['change_'+s+'_'+kind for s in SIGNALS for kind in ['mean_departure','sd_departure','trend_per_week']]
    for name in names:d[name]=np.nan
    audits=[]
    for i,r in d.iterrows():
        recent_start=r.date-pd.Timedelta(days=15);recent_end=r.date-pd.Timedelta(days=8);reference_start=recent_start-pd.Timedelta(days=28)
        person=daily[daily.person.eq(r.person)];recent=person[person.day.ge(recent_start)&person.day.lt(recent_end)];reference=person[person.day.ge(reference_start)&person.day.lt(recent_start)]
        eligible=len(recent)>=5 and len(reference)>=14
        record={'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'recent_start':str(recent_start.date()),'recent_end_exclusive':str(recent_end.date()),'reference_start':str(reference_start.date()),'reference_end_exclusive':str(recent_start.date()),'recent_good_days':len(recent),'reference_good_days':len(reference),'personal_environment_eligible':eligible,'recent_dates':','.join(recent.day.dt.strftime('%Y-%m-%d')),'reference_dates':','.join(reference.day.dt.strftime('%Y-%m-%d'))}
        if len(recent):assert recent.day.min()>=recent_start and recent.day.max()<recent_end<r.date-pd.Timedelta(days=7)
        if len(reference):assert reference.day.min()>=reference_start and reference.day.max()<recent_start
        if eligible:
            x=(recent.day-recent_start).dt.days.to_numpy(float);xc=x-x.mean();assert xc@xc>0
            for signal in SIGNALS:
                mu=reference[signal].mean();sd=reference[signal].std(ddof=0);yy=recent[signal].to_numpy(float)
                d.loc[i,'change_'+signal+'_mean_departure']=recent[signal].mean()-mu
                d.loc[i,'change_'+signal+'_sd_departure']=recent[signal].std(ddof=0)-sd
                d.loc[i,'change_'+signal+'_trend_per_week']=float(xc@(yy-yy.mean())/(xc@xc)*7)
                record['reference_'+signal+'_mean']=mu;record['reference_'+signal+'_sd']=sd
                np.testing.assert_allclose(r['lag_env_'+signal+'_mean'],recent[signal].mean(),atol=1e-10)
                np.testing.assert_allclose(r['lag_env_'+signal+'_sd'],recent[signal].std(ddof=0),atol=1e-10)
        audits.append(record)
    a=pd.DataFrame(audits);d['personal_environment_eligible']=a.personal_environment_eligible.to_numpy();a.to_csv(OUT/'personal_reference_audit.csv',index=False);d.to_csv(OUT/'modeling_features.csv',index=False)
    metadata={'cleaned_reports':len(d),'cleaned_ids':int(d.person.nunique()),'recent_environment_covered_reports':int(d.environment_covered.sum()),'personal_reference_qualified_reports':int(d.personal_environment_eligible.sum()),'personal_reference_qualified_ids':int(d.loc[d.personal_environment_eligible,'person'].nunique()),'daily_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cleaned_stress_sha256':hashlib.sha256((core.OUT/'cleaned_stress_reports.csv').read_bytes()).hexdigest(),'changes':names}
    return d,daily,names,metadata

def evaluate(d,names):
    earlier=pd.read_csv(core.OUT/'environment_comparison_predictions.csv',float_precision='round_trip');records=[];models_saved={}
    env=['lag_env_'+s+'_'+kind for s in SIGNALS for kind in ['mean','sd']]
    scopes={'History + changes':old.HISTORY+names,'History + environment + changes':old.HISTORY+env+names};h=d.copy();splits=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        for block,cutoff,tr,te in old.prev.get_splits(d,protocol):
            train=old.add_history(d.iloc[tr],h);test=old.add_history(d.iloc[te],h,cutoff);train=train[train.prior_report_count.ge(2)];test=test[test.prior_report_count.ge(2)]
            if len(train)<20 or train.person.nunique()<5 or not len(test) or train.excess_stress.ge(.5).nunique()<2:continue
            splits.append({'protocol':protocol,'block':block,'cutoff_exclusive':str(cutoff.date()),'train_n':len(train),'train_changes_measured':int(train.personal_environment_eligible.sum()),'test_n':len(test),'test_changes_measured':int(test.personal_environment_eligible.sum())})
            for scope,nums in scopes.items():
                for forest,family in [(False,'linear'),(True,'forest')]:
                    classifier=old.build(nums,[],forest,True).fit(train[nums],train.excess_stress.ge(.5).astype(int));regressor=old.build(nums,[],forest,False).fit(train[nums],train.excess_stress)
                    for task,preds in [('classification',classifier.predict_proba(test[nums])[:,1]),('regression',np.clip(regressor.predict(test[nums]),-test.past_mean_stress.to_numpy(),4-test.past_mean_stress.to_numpy()))]:
                        for (_,r),pred in zip(test.iterrows(),preds):records.append({'protocol':protocol,'block':block,'task':task,'model':scope+' | '+family,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(r.excess_stress>=.5) if task=='classification' else float(r.excess_stress),'predicted':float(pred),'history_mean':float(r.past_mean_stress),'observed_PSS':float(r.PSS)})
        print('Completed changes',protocol,flush=True)
    p=pd.concat([earlier.drop(columns=['training_pool','environment_covered']),pd.DataFrame(records)],ignore_index=True);p['personal_environment_eligible']=p.source_row.map(d.set_index('source_row').personal_environment_eligible)
    assert not p.duplicated(['protocol','task','model','source_row']).any();p.to_csv(OUT/'predictions.csv',index=False)
    scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={}
        for subset,sub in [('personal_environment_measured',pp[pp.personal_environment_eligible]),('all_test_reports',pp)]:
            scores[protocol][subset]={};comparisons[protocol][subset]={}
            for task,g in sub.groupby('task'):
                scores[protocol][subset][task]={};comparisons[protocol][subset][task]={};keysets=[]
                for name,mm in g.groupby('model'):
                    keysets.append(set(mm.source_row));s=old.classifier_score(mm.observed,mm.predicted) if task=='classification' else old.base.score(mm.observed.to_numpy(),mm.predicted.to_numpy());s['people']=int(mm.person.nunique());scores[protocol][subset][task][name]=s
                assert all(x==keysets[0] for x in keysets)
                for family in ['linear','forest']:
                    for added,baseline in [('History + environment + changes','History + environment'),('History + changes','History'),('History + changes','History + environment')]:
                        aa=g[g.model.eq(added+' | '+family)];bb=g[g.model.eq(baseline+' | '+family)];np.testing.assert_allclose(aa.set_index('source_row').observed,bb.set_index('source_row').loc[aa.source_row].observed,atol=1e-12)
                        comparisons[protocol][subset][task][added+' | '+family+' versus '+baseline+' | '+family]=old.paired_ci(aa,bb,task=='classification')
    full=old.add_history(d,h);full=full[full.prior_report_count.ge(2)]
    for scope,nums in scopes.items():
        for forest,family in [(False,'linear'),(True,'forest')]:
            for classification,task in [(True,'classification'),(False,'regression')]:
                y=full.excess_stress.ge(.5).astype(int) if classification else full.excess_stress;models_saved[scope+' | '+family+' | '+task]={'features':nums,'model':old.build(nums,[],forest,classification).fit(full[nums],y)}
    joblib.dump({'models':models_saved,'plan':PLAN},OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    for name,v in models_saved.items():np.testing.assert_allclose(v['model'].predict(full[v['features']]),restored['models'][name]['model'].predict(full[v['features']]))
    return scores,comparisons,splits

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Personal environmental changes and stress</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#253243}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}table{border-collapse:collapse;width:100%;font-size:14px}</style><h1>Do changes from a student’s usual environment predict above-usual stress?</h1>'
    text+='<p>Compare departures from an earlier personal baseline, recent environmental trends and unusual variability with the previous history/environment models. The target and all validation dates/settings are unchanged. The primary comparison uses identical reports with both recent and earlier reference sensor coverage.</p><h2>Coverage</h2><pre>'+html.escape(json.dumps(r['metadata'],indent=2))+'</pre>'
    models=r['scores']['purged_rolling_blocks']['personal_environment_measured'];before=models['classification']['History + environment | forest'];after=models['classification']['History + environment + changes | forest'];reg_before=models['regression']['History + environment | forest'];reg_after=models['regression']['History + environment + changes | forest']
    text+=f'<h2>Result</h2><p><strong>No demonstrated predictive gain from personalized environmental changes.</strong> The common rolling test includes{before["n"]} reports from{before["people"]} IDs and only{before["events"]} above-usual weeks. Adding changes to the existing environmental forest changes excess-stress MAE from{reg_before["mae"]:.4f} to{reg_after["mae"]:.4f}, and classification Brier from{before["brier"]:.4f} to{after["brier"]:.4f} (higher is worse for both). The corresponding paired improvement intervals include zero. Replacing environmental levels with personalized changes also does not establish improvement over history alone. The fixed linear regression becomes worse with these additional features.</p><p>On this exact subset, the environmental logistic classifier AUC changes0.713→0.648 when changes are added, and forest AUC0.566→0.520. All new models detect none of the11 rolling events at the unchanged0.5 probability cutoff; logistic variants generate false alerts. The separate April test also fails to demonstrate a gain. These results do not show that room conditions have no relationship to stress; baseline coverage is sparse and only11 rolling events remain.</p><p>The forest MAE near0.334 here must not be compared directly with0.393 from the earlier984-report population: this is a different, more selectively measured138-report group. Each model comparison below retains identical test rows. Reference windows and feature settings were fixed before this run; no alternatives were selected for better test scores.</p>'
    for protocol,subsets in r['scores'].items():
        for subset,tasks in subsets.items():
            text+='<h2>'+html.escape(protocol+' / '+subset)+'</h2><table><tr><th>Classifier</th><th>N / IDs / events</th><th>AUC</th><th>AP</th><th>Brier↓</th><th>True / false alerts</th><th>Recall</th></tr>'
            for name,s in tasks['classification'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{s["n"]} / {s["people"]} / {s["events"]}</td><td>{s["roc_auc"]:.3f}</td><td>{s["average_precision"]:.3f}</td><td>{s["brier"]:.4f}</td><td>{s["tp"]} / {s["fp"]}</td><td>{s["recall"]:.3f}</td></tr>'
            text+='</table><table><tr><th>Regressor</th><th>MAE↓</th><th>RMSE↓</th></tr>'
            for name,s in tasks['regression'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{s["mae"]:.3f}</td><td>{s["rmse"]:.3f}</td></tr>'
            text+='</table><table><tr><th>Task / comparison</th><th>Improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
            for task,cs in r['comparisons'][protocol][subset].items():
                for name,v in cs.items():
                    lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(task+" / "+name)}</td><td>{v["improvement"]:.4f}</td><td>{lo:.4f} to {hi:.4f}</td></tr>'
            text+='</table>'
    text+='<h2>Definitions and limitations</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p>Classification decision threshold remains0.5. No feature windows, baselines or alert threshold chosen to maximize test performance. The same study has been repeatedly explored; all findings require independent confirmation. Calendar sensors do not establish actual individual exposure, and statistical association/prediction does not show causation.</p><p><a href="results.json">Full results</a> · <a href="analysis_plan.json">Definitions</a> · <a href="personal_reference_audit.csv">Personal-reference and timing audit</a> · <a href="predictions.csv">Predictions</a> · <a href="independent_checks.json">Independent checks</a></p>'
    (OUT/'report.html').write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));d,daily,names,meta=prepare();print('Coverage',json.dumps(meta),flush=True);scores,comparisons,splits=evaluate(d,names)
    checks={'references_use_only_days_strictly_before_recent_window':True,'recent_and_reference_windows_do_not_overlap':True,'existing_environment_features_reproduce_daily_aggregates':True,'test_histories_frozen_as_previously':True,'all_comparisons_share_test_rows':True,'train_only_preprocessing':True,'serialized_models_roundtrip':True}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2));r={'plan':PLAN,'metadata':meta,'scores':scores,'comparisons':comparisons,'splits':splits};(OUT/'results.json').write_text(json.dumps(r,indent=2));render(r);print(json.dumps({'scores':scores,'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
