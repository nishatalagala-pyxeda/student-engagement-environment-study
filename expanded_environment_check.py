"""Earlier environment addition to the expanded stress-history models."""
from pathlib import Path
import json,html,hashlib
import numpy as np
import pandas as pd
import expanded_stress_increases as core
old=core.old;OUT=core.OUT
PLAN={'question':'Do earlier environmental measurements add predictive value to the expanded history model?','windows':'Same earlier environment features as previous study: [D-15,D-8), at least5 good days (72 distinct15-minute slots/day). Reuse independently checked daily aggregates. No target-week sensor values.','comparison':'Same full-report training pools/calendar blocks and target references as expanded history test. History alone versus history+14 earlier environmental means/SDs, identical fixed linear and forest families. Missing environment imputed using training only; main effect comparison on test rows with measured environment, alongside all-row results. Models trained on all history-eligible earlier reports.','limits':'Environmental coverage is selected and sparse; imputation/availability patterns may predict independently of exposure. Sensor mapping inferred, F135 excluded. Exploratory, no causal claims or new alert threshold.'}

def prepare():
    d=pd.read_csv(OUT/'cleaned_stress_reports.csv');d.date=pd.to_datetime(d.date);daily=pd.read_csv(core.ROOT/'outputs/stress-increases/environment_daily.csv');daily.day=pd.to_datetime(daily.day)
    names=['lag_env_'+c+'_'+s for c in old.SIGNALS for s in ['mean','sd']]
    for c in names:d[c]=np.nan
    rows=[]
    for i,r in d.iterrows():
        start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8);ds=daily[daily.person.eq(r.person)&daily.day.ge(start)&daily.day.lt(end)]
        if len(ds)>=5:
            for c in old.SIGNALS:d.loc[i,'lag_env_'+c+'_mean']=ds[c].mean();d.loc[i,'lag_env_'+c+'_sd']=ds[c].std(ddof=0)
        rows.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'start':str(start.date()),'end_exclusive':str(end.date()),'good_days':len(ds)})
    d['environment_covered']=d[names].notna().all(axis=1);d.to_csv(OUT/'environment_modeling_features.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'environment_feature_audit.csv',index=False)
    return d,names

def main():
    (OUT/'environment_analysis_plan.json').write_text(json.dumps(PLAN,indent=2));d,names=prepare();baseline=pd.read_csv(OUT/'expanded_predictions.csv',float_precision='round_trip');h=d.copy();records=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        for block,cutoff,tr,te in old.prev.get_splits(d,protocol):
            train=old.add_history(d.iloc[tr],h);test=old.add_history(d.iloc[te],h,cutoff);train=train[train.prior_report_count.ge(2)];test=test[test.prior_report_count.ge(2)]
            if len(train)<20 or train.person.nunique()<5 or not len(test) or train.excess_stress.ge(.5).nunique()<2:continue
            for forest,family in [(False,'linear'),(True,'forest')]:
                ns=old.HISTORY+names;classifier=old.build(ns,[],forest,True).fit(train[ns],train.excess_stress.ge(.5).astype(int));regressor=old.build(ns,[],forest,False).fit(train[ns],train.excess_stress)
                for task,preds in [('classification',classifier.predict_proba(test[ns])[:,1]),('regression',np.clip(regressor.predict(test[ns]),-test.past_mean_stress.to_numpy(),4-test.past_mean_stress.to_numpy()))]:
                    for (_,r),pred in zip(test.iterrows(),preds):records.append({'protocol':protocol,'block':block,'training_pool':'Expanded training','task':task,'model':'History + environment | '+family,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(r.excess_stress>=.5) if task=='classification' else float(r.excess_stress),'predicted':float(pred),'history_mean':float(r.past_mean_stress),'observed_PSS':float(r.PSS)})
        print('Environment done',protocol,flush=True)
    added=pd.DataFrame(records);p=pd.concat([baseline[baseline.model.str.startswith('History')],added],ignore_index=True);coverage=d.set_index('source_row').environment_covered;p['environment_covered']=p.source_row.map(coverage)
    p.to_csv(OUT/'environment_comparison_predictions.csv',index=False);scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={}
        for subset,sub in [('measured_earlier_environment',pp[pp.environment_covered]),('all_test_reports',pp)]:
            scores[protocol][subset]={};comparisons[protocol][subset]={}
            for task,g in sub.groupby('task'):
                scores[protocol][subset][task]={}
                for name,mm in g.groupby('model'):
                    value=old.classifier_score(mm.observed,mm.predicted) if task=='classification' else old.base.score(mm.observed.to_numpy(),mm.predicted.to_numpy());value['people']=int(mm.person.nunique());scores[protocol][subset][task][name]=value
                comparisons[protocol][subset][task]={}
                for family in ['linear','forest']:
                    a=g[g.model.eq('History + environment | '+family)];b=g[g.model.eq('History | '+family)];aa=a.set_index('source_row');bb=b.set_index('source_row').loc[aa.index]
                    assert len(a)==len(b) and set(a.source_row)==set(b.source_row);np.testing.assert_allclose(aa.observed,bb.observed,atol=1e-12)
                    comparisons[protocol][subset][task][family]=old.paired_ci(a,b,task=='classification')
    result={'plan':PLAN,'source_daily_sha256':hashlib.sha256((core.ROOT/'outputs/stress-increases/environment_daily.csv').read_bytes()).hexdigest(),'covered_cleaned_reports':int(d.environment_covered.sum()),'covered_ids':int(d.loc[d.environment_covered,'person'].nunique()),'scores':scores,'comparisons':comparisons}
    (OUT/'environment_results.json').write_text(json.dumps(result,indent=2))
    text='<h2>Environmental additions with the larger training pool</h2><p>Additional comparison: models use all earlier history-eligible reports for training, with or without14 earlier environmental features. Main comparison retains the identical test rows with measured earlier environment. Missing measurements in training use training-only imputation. No demographics, Fitbit or target-week environment enter. These are selected observed-bedroom measurements with inferred ID mapping, so predictive differences do not establish a causal effect.</p>'
    for protocol,subsets in scores.items():
        for subset,tasks in subsets.items():
            text+='<h3>'+html.escape(protocol+' / '+subset)+'</h3><table><tr><th>Classifier</th><th>N / IDs / events</th><th>AUC</th><th>Brier↓</th><th>True / false alerts</th></tr>'
            for name,v in tasks['classification'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["n"]} / {v["people"]} / {v["events"]}</td><td>{v["roc_auc"]:.3f}</td><td>{v["brier"]:.4f}</td><td>{v["tp"]} / {v["fp"]}</td></tr>'
            text+='</table><table><tr><th>Regressor</th><th>MAE↓</th></tr>'
            for name,v in tasks['regression'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae"]:.3f}</td></tr>'
            text+='</table><table><tr><th>Task / family: environment addition</th><th>Improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
            for task,fams in comparisons[protocol][subset].items():
                for family,v in fams.items():
                    lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{task} / {family}</td><td>{v["improvement"]:.4f}</td><td>{lo:.4f} to {hi:.4f}</td></tr>'
            text+='</table>'
    text+='<p><a href="environment_results.json">Full environmental results</a> · <a href="environment_feature_audit.csv">Earlier window audit</a></p>'
    (OUT/'environment_report_section.html').write_text(text)
    core.render(json.loads((OUT/'results.json').read_text()))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
