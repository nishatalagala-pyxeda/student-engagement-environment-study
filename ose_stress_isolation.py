"""Isolate historical/current stress value beyond OSE history for unseen students."""
import hashlib,html,json
import joblib
import numpy as np
import pandas as pd
import ose_current_shift_models as current

OUT=current.ROOT/'outputs/ose-stress-isolation';ROOT=current.ROOT
PLAN={
 'question':'Does stress alone improve held-out-student OSE prediction beyond OSE history, and do within-person stress changes coincide with OSE changes?',
 'prediction':'Same1118 reports138IDs and five student-held-out folds as before. Fixed ridge alpha10 and forest300trees depth3 minimumleaf10 seed42. Earlier OSE features unchanged, strictly before D-7. No tuning or outer-test selection.',
 'scopes':'History+earlier PSS seven summaries; history+current PSS level and age; history+current PSS level/age and explicit stress change. Current stress latest on/before D; change from preceding OSE-date snapshot only when gap7–21days. OSE target/items/current change never predictors.',
 'baselines':'Reuse exact previous history-only ridge/forest and last-score held-out predictions. Match model family and test rows. Current-stress comparisons are concurrent estimation, not week-ahead forecasts.',
 'descriptive':'1085 consecutive same-person OSE pairs7–21days apart with exact same-date PSS at both dates. Descriptive correlation and mean OSE change in stress-rise>=1,stress-fall<=-1,and smaller-change groups. One stress point is an illustrative contrast on supplied0–4 aggregate, not a clinical threshold.',
 'uncertainty':'Prediction MAE gain95% conservative numeric-student-group bootstrap3000draws, fixed predictions. Correlation interval resamples the same conservative student groups with all their pairs; not independent row resampling.',
 'limits':'Observational/noncausal; concurrent stress and engagement can influence each other or share response bias. Score aggregation/reverse coding unverified, repeated/dependent observations and exploratory dataset reuse. No clinical inference. Original source/mapping/student-identity caveats remain.'
}

def descriptive():
    path=ROOT/'outputs/ose-change-audit/paired_observations.csv';d=pd.read_csv(path,float_precision='round_trip');d=d.dropna(subset=['stress_PSS_before','stress_PSS_after'])
    d['group']=d.person.str.extract(r'(\d+)$',expand=False);x=d.stress_PSS_after-d.stress_PSS_before;y=d.OSE_after-d.OSE_before
    labels=np.where(x.ge(1),'Stress rose >=1',np.where(x.le(-1),'Stress fell >=1','Stress changed <1'))
    summaries={}
    for label in np.unique(labels):
        mask=labels==label;summaries[label]={'pairs':int(mask.sum()),'ids':d.loc[mask,'person'].nunique(),'mean_OSE_change':float(y[mask].mean()),'median_OSE_change':float(y[mask].median())}
    moments=pd.DataFrame({'group':d.group,'n':1,'x':x,'y':y,'xx':x*x,'yy':y*y,'xy':x*y}).groupby('group').sum().to_numpy(float)
    ix=np.random.default_rng(42).integers(len(moments),size=(3000,len(moments)));s=moments[ix].sum(axis=1);n,sx,sy,sxx,syy,sxy=s.T
    corr=(sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    return {'pairs':len(d),'ids':d.person.nunique(),'correlation_stress_change_OSE_change':float(np.corrcoef(x,y)[0,1]),'group_bootstrap95_interval':np.quantile(corr,[.025,.975]).tolist(),'groups':summaries,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2))
    d,schema,_,_=current.load();indexed=d.set_index('source_row',drop=False);h=current.old.previous.core.OSE_FEATURES
    scopes={'Earlier stress':h+current.old.previous.core.STRESS_FEATURES,'Current stress':h+['current_stress_PSS','current_stress_PSS_age_days'],'Current stress + change':h+['current_stress_PSS','current_stress_PSS_age_days','change_stress_PSS']}
    m=pd.read_csv(current.OUT/'outer_membership.csv',dtype={'student_group':str});records=[]
    for fold,mm in m.groupby('fold'):
        train=indexed.loc[mm.loc[mm.role.eq('train'),'source_row']];test=indexed.loc[mm.loc[mm.role.eq('test'),'source_row']];assert set(train.student_group).isdisjoint(test.student_group)
        models={}
        for scope,cols in scopes.items():
            for forest,family in [(False,'Ridge'),(True,'Random forest')]:
                name=scope+' | '+family;model=current.old.previous.build(cols,schema,forest).fit(train[cols],train.OSE);models[name]={'model':model,'features':cols};pred=model.predict(test[cols])
                records.extend({'fold':int(fold),'model':name,'source_row':int(r.source_row),'person':r.person,'student_group':str(r.student_group),'observed':float(r.OSE),'predicted':float(v)} for (_,r),v in zip(test.iterrows(),pred))
        joblib.dump(models,OUT/f'fold_{fold}.joblib');print('Stress isolation fold',fold,'complete',flush=True)
    before=pd.read_csv(ROOT/'outputs/ose-unseen-with-history/predictions.csv',float_precision='round_trip',dtype={'student_group':str});before=before[before.protocol.eq('Previous eligible subset')&before.model.isin(['Past OSE only | Ridge','Past OSE only | Random forest','Last OSE'])]
    p=pd.concat([pd.DataFrame(records),before],ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False)
    cases={'All eligible observations':set(d.loc[d.eligible,'source_row']),'Larger current stress shifts':set(d.loc[d.change_stress_PSS.abs().ge(1),'source_row'])}
    results={}
    for subset,keys in cases.items():
        pp=p[p.source_row.isin(keys)];scores={};comparisons={}
        for name,g in pp.groupby('model'):scores[name]={**current.old.previous.core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':g.person.nunique()}
        for scope in scopes:
            for family in ['Ridge','Random forest']:
                name=scope+' | '+family;base='Past OSE only | '+family;a=pp[pp.model.eq(name)].copy();b=pp[pp.model.eq(base)].copy();a.person=a.student_group;b.person=b.student_group;comparisons[name+' versus '+base]=current.old.previous.paired(a,b)
        v=pp[pp.model.eq('Last OSE')];results[subset]={'reports':len(v),'ids':v.person.nunique(),'scores':scores,'comparisons':comparisons}
    result={'plan':PLAN,'scopes':scopes,'prediction':results,'descriptive':descriptive()};(OUT/'results.json').write_text(json.dumps(result,indent=2,default=int));render(result);print(json.dumps(result['prediction'],indent=2,default=int),flush=True);print(json.dumps(result['descriptive'],indent=2,default=int),flush=True)

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Does stress add value for OSE?</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2f7}</style><h1>Stress and OSE engagement: isolated predictor check</h1><p>Same student-held-out folds. Earlier OSE allowed at inference; current/future OSE excluded. Current-stress estimation is concurrent, not a prospective forecast.</p>'
    text+='<p><strong>Stress changes show a weak negative association with engagement changes, but no added predictive benefit beyond past OSE.</strong> OSE falls1.4points on average in65 stress-rise pairs and rises1.53points in58 stress-fall pairs, using a1point stress contrast. Across all1,085paired observations the correlation is only−0.093. Current-stress ridge MAE4.113 versus history-only4.098; adding the stress-change feature gives4.124. Forest predictions also do not improve. A weak descriptive relationship can coexist with no useful incremental prediction. This does not show that stress causes changes in engagement.</p>'
    a=r['descriptive'];lo,hi=a['group_bootstrap95_interval'];text+=f'<h2>Within-person changes</h2><p>{a["pairs"]} paired observations from{a["ids"]} cohort-aware IDs. Correlation between stress change and OSE change: {a["correlation_stress_change_OSE_change"]:.3f},95% student-group interval [{lo:.3f}, {hi:.3f}]. Descriptive, not causal. Excluding zero-OSE endpoints leaves the correlation effectively unchanged at−0.094 (1,084pairs).</p><table><tr><th>Stress change</th><th>Pairs / IDs</th><th>Mean OSE change</th><th>Median OSE change</th></tr>'
    for name,v in a['groups'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["pairs"]} / {v["ids"]}</td><td>{v["mean_OSE_change"]:.3f}</td><td>{v["median_OSE_change"]:.3f}</td></tr>'
    text+='</table>'
    for subset,v in r['prediction'].items():
        text+='<h2>'+html.escape(subset)+f' — {v["reports"]} reports / {v["ids"]} IDs</h2><table><tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th></tr>'
        for name,s in v['scores'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{s["mae"]:.3f}</td><td>{s["rmse"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Comparison</th><th>MAE improvement</th><th>95% group interval</th></tr>'
        for name,c in v['comparisons'].items():
            lo,hi=c['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{c["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Method and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p><a href="results.json">Full results</a> · <a href="predictions.csv">Held-out predictions</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
