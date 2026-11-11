"""Current conditions/change models with nested tuning and unseen-student tests."""
import json,html,time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import ose_current_shift_features as features

OUT=features.OUT;ROOT=features.ROOT;old=features.old
ALPHAS=[.1,1,10]
MODEL_PLAN={
 'algorithms':'Original fixed forest300trees depth3 minimumleaf10 seed42, and Elastic Net l1ratio.9 with alpha selected from.1/1/10 by three inner student-held-out folds. No outer-test parameter selection.',
 'preprocessing':'Every inner imputer/scaler/encoder/facial PCA fitted only inner training students. Final refit uses outer training students. Numeric missing values median+indicators; current/older facial features jointly PCA10. Categorical fields remain train-only one-hot.',
 'scopes':'OSE history only; history+current physical levels/changes; history+current physical and environmental satisfaction levels/changes; all historical+current levels; all historical+current levels and explicit changes.',
 'comparison':'Match algorithm and test records. Primary all current+changes versus history-only on substantial-shift cases; secondary levels-only ablation, environmental-only scopes and overall cases. Previous history-only ridge/last OSE and lagged all-predictor forecasts included as fixed baselines.',
 'uncertainty':'Paired conservative student-group bootstrap3000draws of fixed held-out errors; no refitting or model-selection uncertainty. Multiple exploratory subgroups and reused test data, not confirmatory inference.'
}

def load():
    schema=json.loads((OUT/'feature_schema.json').read_text());metadata=json.loads((OUT/'feature_metadata.json').read_text());d=pd.read_csv(OUT/'modeling_features.csv',float_precision='round_trip',dtype={c:str for c in schema['categorical_features']});d.date=pd.to_datetime(d.date);return d,schema,metadata['scopes'],metadata

def subset_definitions(d):
    events=pd.read_csv(ROOT/'outputs/ose-change-audit/large_change_examples.csv');sets={}
    sets['Physical shifts']=set(events.loc[events.group.eq('physical_recent'),'source_row_after'])
    env=events.factor.str.startswith('satisfaction_');sets['Environmental satisfaction shifts']=set(events.loc[env,'source_row_after'])
    other=events.group.eq('self_report_or_network')&~env;sets['Other factor shifts']=set(events.loc[other,'source_row_after'])
    sets['Any substantial shift']=set.union(*sets.values());sets['All eligible observations']=set(d.loc[d.eligible,'source_row'])
    return sets

def main():
    d,schema,scopes,metadata=load();indexed=d.set_index('source_row',drop=False)
    plan={**features.PLAN,**MODEL_PLAN,'alphas':ALPHAS,'scopes':scopes};(OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    memberships=pd.read_csv(old.OUT/'fold_membership.csv',dtype={'student_group':str});memberships=memberships[memberships.protocol.eq('Previous eligible subset')]
    memberships.to_csv(OUT/'outer_membership.csv',index=False)
    predictions=[];tuning=[];inner_members=[];selected=[]
    for fold,mm in memberships.groupby('fold'):
        started=time.monotonic();train=indexed.loc[mm.loc[mm.role.eq('train'),'source_row']].reset_index(drop=True);test=indexed.loc[mm.loc[mm.role.eq('test'),'source_row']].reset_index(drop=True)
        assert set(train.student_group).isdisjoint(test.student_group);totals={scope:np.zeros(len(ALPHAS)) for scope in scopes}
        for inner,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=142).split(train,groups=train.student_group),1):
            a=train.iloc[it];b=train.iloc[iv];assert set(a.student_group).isdisjoint(b.student_group)
            for role,g in [('train',a),('validation',b)]:inner_members.extend(g[['source_row','person','student_group']].assign(outer_fold=int(fold),inner_fold=inner,role=role).to_dict('records'))
            for scope,cols in scopes.items():
                pre=old.previous.build(cols,schema,False).steps[0][1];pre.fit(a[cols]);x=pre.transform(a[cols]);y=pre.transform(b[cols])
                for i,alpha in enumerate(ALPHAS):
                    model=ElasticNet(alpha=alpha,l1_ratio=.9,max_iter=50000,tol=1e-4,random_state=42).fit(x,a.OSE);pred=model.predict(y);error=float(np.abs(b.OSE.to_numpy()-pred).sum());totals[scope][i]+=error
                    tuning.append({'outer_fold':int(fold),'inner_fold':inner,'scope':scope,'alpha':alpha,'validation_n':len(b),'mae':error/len(b)})
            print('Outer',fold,'inner',inner,'complete',round(time.monotonic()-started,1),'seconds',flush=True)
        bundles={}
        for scope,cols in scopes.items():
            i=int(np.argmin(totals[scope]));alpha=ALPHAS[i];selected.append({'outer_fold':int(fold),'scope':scope,'alpha':alpha,'inner_mae':float(totals[scope][i]/len(train))})
            pre=old.previous.build(cols,schema,False).steps[0][1];pre.fit(train[cols]);x=pre.transform(train[cols]);y=pre.transform(test[cols])
            for family in ['Random forest','Elastic Net']:
                model=RandomForestRegressor(n_estimators=300,max_depth=3,min_samples_leaf=10,random_state=42,n_jobs=1) if family=='Random forest' else ElasticNet(alpha=alpha,l1_ratio=.9,max_iter=50000,tol=1e-4,random_state=42)
                model.fit(x,train.OSE);pred=model.predict(y);assert np.isfinite(pred).all();name=scope+' | '+family
                bundles[name]={'features':cols,'preprocessor':pre,'estimator':model}
                predictions.extend({'fold':int(fold),'model':name,'source_row':int(r.source_row),'person':r.person,'student_group':str(r.student_group),'observed':float(r.OSE),'predicted':float(v)} for (_,r),v in zip(test.iterrows(),pred))
        joblib.dump(bundles,OUT/f'fold_{fold}_models.joblib')
        pd.DataFrame(predictions).to_csv(OUT/'new_predictions_checkpoint.csv',index=False);pd.DataFrame(tuning).to_csv(OUT/'inner_tuning.csv',index=False);pd.DataFrame(inner_members).to_csv(OUT/'inner_membership.csv',index=False);(OUT/'selected_alphas.json').write_text(json.dumps(selected,indent=2))
        print('Outer fold',fold,'complete',round(time.monotonic()-started,1),'seconds',flush=True)
    previous=pd.read_csv(ROOT/'outputs/ose-unseen-with-history/predictions.csv',float_precision='round_trip',dtype={'student_group':str});previous=previous[previous.protocol.eq('Previous eligible subset')]
    keep=['Past OSE only | Ridge','Past OSE only | Random forest','Last OSE','All including past OSE | Random forest','All including past OSE | Ridge']
    previous=previous[previous.model.isin(keep)].copy();previous.model=previous.model.replace({'Past OSE only | Ridge':'Previous history | Ridge','Past OSE only | Random forest':'Previous history | Random forest','All including past OSE | Random forest':'Previous lagged all | Random forest','All including past OSE | Ridge':'Previous lagged all | Ridge'})
    p=pd.concat([pd.DataFrame(predictions),previous],ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False)
    definitions=subset_definitions(d);rows=[{'subset':name,'source_row':int(key)} for name,keys in definitions.items() for key in sorted(keys)];pd.DataFrame(rows).to_csv(OUT/'subgroup_membership.csv',index=False)
    results={}
    for subset,keys in definitions.items():
        pp=p[p.source_row.isin(keys)];scores={};comparisons={}
        for name,g in pp.groupby('model'):
            errors=abs(g.observed-g.predicted);scores[name]={**old.previous.core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':g.person.nunique(),'equal_student_mae':float(errors.groupby(g.person).mean().mean())}
        for family in ['Random forest','Elastic Net']:
            for scope in scopes:
                if scope=='History only':continue
                a=pp[pp.model.eq(scope+' | '+family)].copy()
                for base in ['History only | '+family,'Previous history | Ridge','Last OSE']+(['All current levels | '+family] if scope=='All current + changes' else []):
                    b=pp[pp.model.eq(base)].copy();a.person=a.student_group;b.person=b.student_group
                    comparisons[scope+' | '+family+' versus '+base]=old.previous.paired(a,b)
        coverage=pp[pp.model.eq('Last OSE')];results[subset]={'reports':len(coverage),'ids':coverage.person.nunique(),'scores':scores,'comparisons':comparisons}
    result={'plan':plan,'subsets':results,'feature_source_hashes':metadata['source_hashes']};(OUT/'results.json').write_text(json.dumps(result,indent=2,default=int));render(result)
    print(json.dumps({name:{'reports':v['reports'],'ids':v['ids'],'scores':{model:score['mae'] for model,score in v['scores'].items()}} for name,v in results.items()},indent=2,default=int),flush=True)

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Current conditions and shifts for unseen-student OSE</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}</style><h1>Current conditions and explicit changes: OSE for unseen students</h1><p>Same five student-held-out folds and1,118 history-eligible reports. Earlier OSE is allowed, current/future OSE excluded. New non-OSE measurements are supplied around the target date. This is concurrent assisted estimation, not week-ahead forecasting. Learning-impact questionnaire items are excluded from the current block.</p>'
    text+='<h2>Main comparison: same tuned Elastic Net algorithm</h2><table><tr><th>Shift cases</th><th>Reports / IDs</th><th>OSE history only MAE</th><th>All current predictors + changes MAE</th><th>MAE improvement /95% interval</th></tr>'
    for subset in ['Physical shifts','Environmental satisfaction shifts','Other factor shifts','Any substantial shift']:
        v=r['subsets'][subset];s=v['scores'];c=v['comparisons']['All current + changes | Elastic Net versus History only | Elastic Net'];lo,hi=c['person_cluster_95_interval']
        text+=f'<tr><td>{html.escape(subset)}</td><td>{v["reports"]} / {v["ids"]}</td><td>{s["History only | Elastic Net"]["mae"]:.2f}</td><td>{s["All current + changes | Elastic Net"]["mae"]:.2f}</td><td>{c["mae_improvement"]:.2f} / [{lo:.2f}, {hi:.2f}]</td></tr>'
    text+='</table><p><strong>No reliable incremental gain was demonstrated.</strong> The physical-shift subgroup shows an apparent0.20point Elastic Net gain, but its interval includes no gain and worsening. It has only61 reports26 student IDs. Across697 reports135IDs with any substantial shift, the all-predictor Elastic Net is slightly worse than history-only. The forest comparison also provides no reliable gain. Adding explicit changes to current levels does not improve the models: Elastic Net predictions are effectively identical and forest differences negligible. The current environmental-only models do not establish a reliable gain either. These are exploratory internal results and do not establish absence of an environmental effect.</p><p>Shift subgroups overlap. The physical/satisfaction/other union here is contemporaneous697 observations; it is different from the698 earlier-known-shift observations in the preceding forecast analysis.</p>'
    for subset,v in r['subsets'].items():
        text+='<h2>'+html.escape(subset)+f' — {v["reports"]} reports / {v["ids"]} IDs</h2><table><tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>R²</th></tr>'
        for name,s in v['scores'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{s["mae"]:.3f}</td><td>{s["rmse"]:.3f}</td><td>{s["r2"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>95% group-bootstrap interval</th></tr>'
        for name,c in v['comparisons'].items():
            lo,hi=c['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{c["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Methods and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(str(v))+'</p>' for k,v in r['plan'].items())
    text+='<p><a href="results.json">Full results</a> · <a href="feature_metadata.json">Feature definitions</a> · <a href="selected_alphas.json">Inner-selected regularization</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
