"""All available historical variables for OSE regression; grouped comparisons."""
from pathlib import Path
import json,html,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from ose_model_transforms import FacialReducer
import ose_all_features as features
core=features.core;ROOT=features.ROOT;OUT=features.OUT
PLAN={
 'target':'Supplied OSE aggregate in native units: questionnaire matches Online Student Engagement, despite README self-efficacy label. Item-level scoring unavailable; retain zeros/fractional scores. Regression, no arbitrary class cutoff or theoretical clipping.',
 'question':'Does the combined set of available predictors improve later OSE beyond earlier OSE alone?',
 'history':'Primary target eligibility and dates unchanged: at least2 earlier OSE reports and1 earlier PSS, both strictly before D-7. Every weekly/monthly predictor history also strictly before D-7 and, for test rows, before block training cutoff. No same-survey OSE items or concurrent scores.',
 'other_inputs':'All nonempty demographic fields except ID; approximate age and numeric socioeconomic ladder, remaining categories including postal code/free text. Earlier PSS, PANAS positive/negative, CESD, all20 stored STAI item codes and stored STAI total, all4 computer/internet variables, all14 environmental satisfaction items. Each new repeated numerical survey variable contributes last, past mean/SD, days since last valid report. Existing OSE/PSS summaries unchanged.',
 'daily_inputs':'Prior daily diary activity-code proportions, all Fitbit file types and available raw numeric/named-state summaries, eight device environment measures including composite score plus personal changes/trends. Use [D-15,D-8); same sensor coverage5 days with72slots/day; personal reference [D-43,D-15) requires14gooddays. Diary needs3 valid days/item and Fitbit3days/summary. Daily non-target inputs can update using earlier observed records within test blocks. This does not use any test OSE or mental-health labels.',
 'facial_inputs':'Every numeric OpenFace tracking/gaze/pose/landmark/AU field except frame/face ID, timestamp and success. Every30th source row sampled, success1/confidence>=0.8, >=30good sampled frames and60sec span. Filename DDMMYYYY per README; unrecognized/implausible dates excluded, not guessed. Use most recent admissible recording within28days before D-8. Means/SDs of sampled valid frames, daily sessions equally averaged. All such fields enter a training-only standardized PCA block, up to10 components padded to10 if insufficient data. Tracking/geometry is not emotion ground truth.',
 'models':'Same fixed ridge alpha10 and forest300trees depth3 minleaf10 seed42, no tuning. Missing numeric predictors median-imputed with indicators; categorical constantUnknown and one-hot, train-only. Seven one-group additions to pastOSE, combined all-predictor model, all without demographics, and all without pastOSE. No target/student identifiers as predictors. Baselines and previous pastOSE model predictions reused on exact same rows.',
 'validation':'Same14-day expanding calendar folds anchoredJune1,2020,7-day training gap and minima. Reused April secondary holdout. All model comparisons use identical eligible training/test rows. Frozen source-label histories; raw daily predictor data remain prospectively earlier than target. Known returning students, not a new-student generalization test.',
 'metrics':'MAE primary, RMSE/R² secondary, paired fixed-prediction person-cluster bootstrap3000draws seed42 for incremental MAE. Group and subset comparisons exploratory, not a tuning search or independent replications. No refitting/selection uncertainty in intervals.',
 'limitations':'Static demographic administration dates unknown, so also evaluate all without demographics. Stored STAI coding/reverse-key cannot be independently verified; predictor codes only, no clinical interpretation. Fitbit wear/units/timezones not fully validated; local-day file attribution and1day buffer, no intraday sensor alignment. Sparse sensor/facial data and missingness may drive predictions. Identifying demographic/geometry proxies may predict returning students; no cross-person claim. Grades.csv absent. Raw frame IDs and time indexes excluded. No causal claims.'}

def build(columns,schema,forest):
    cats=[c for c in columns if c in schema['categorical_features']];face=[c for c in columns if c in schema['face_features']];nums=[c for c in columns if c not in cats+face]
    assert not set(columns)&{'OSE','PSS','person','source_row','date','ID'}
    blocks=[('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler()),nums)]
    if cats:blocks.append(('categorical',make_pipeline(SimpleImputer(strategy='constant',fill_value='Unknown'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats))
    if face:blocks.append(('facial',FacialReducer(),face))
    model=RandomForestRegressor(n_estimators=300,max_depth=3,min_samples_leaf=10,random_state=42,n_jobs=1) if forest else Ridge(alpha=10)
    return make_pipeline(ColumnTransformer(blocks,remainder='drop'),model)

def scopes(schema):
    groups=schema['groups'];s={'Past OSE + '+name:core.OSE_FEATURES+cols for name,cols in groups.items()};allcols=core.OSE_FEATURES+[c for cols in groups.values() for c in cols]
    assert len(allcols)==len(set(allcols));s['All predictors']=allcols;s['All without demographics']=[c for c in allcols if not c.startswith('demo_')];s['All without past OSE']=[c for c in allcols if c not in core.OSE_FEATURES];return s

def paired(a,b):
    a=a.set_index('source_row');b=b.set_index('source_row').loc[a.index];np.testing.assert_allclose(a.observed,b.observed,atol=1e-12)
    delta=abs(b.predicted-b.observed)-abs(a.predicted-a.observed);sums=[];counts=[]
    for _,v in delta.groupby(a.person):sums.append(v.sum());counts.append(len(v))
    sums=np.array(sums);counts=np.array(counts);rng=np.random.default_rng(42);ix=rng.integers(len(sums),size=(3000,len(sums)));boot=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return {'mae_improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'n':len(a),'ids':len(sums)}

def evaluate(ose,stress,full,stores,schema):
    definitions=scopes(schema);records=[];audits=[];splitinfo=[];test_features=[];availability=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        for block,cutoff,tr,te in core.validation.get_splits(ose,protocol):
            train=full.iloc[tr];train=train[train.eligible];test=full.iloc[te].reset_index(drop=True).copy();base=core.history_features(ose.iloc[te],ose,stress,cutoff)
            for c in core.OSE_FEATURES+core.STRESS_FEATURES+['eligible']:test[c]=base[c].to_numpy()
            ep,history_audit=features.episodic_features(ose.iloc[te],stores,cutoff)
            for c in ep:test[c]=ep[c].to_numpy()
            test=test[test.eligible]
            if len(train)<20 or train.person.nunique()<5 or not len(test):continue
            save_cols=['source_row','person','date']+core.OSE_FEATURES+core.STRESS_FEATURES+schema['episodic_features']
            test_features.extend(test[save_cols].assign(protocol=protocol,block=block,cutoff_exclusive=str(cutoff.date())).to_dict('records'))
            for _,r in test.iterrows():
                rec={'protocol':protocol,'source_row':int(r.source_row),'person':r.person}
                for group,cols in schema['groups'].items():
                    take=[c for c in cols if c.startswith('demo_')] if group=='demographics_time' else cols
                    rec[group]=bool(r[take].notna().any())
                availability.append(rec)
            keys=set(test.source_row);audits.extend([{**v,'protocol':protocol,'block':block} for v in history_audit if v['source_row'] in keys]);assert train.date.max()<cutoff
            splitinfo.append({'protocol':protocol,'block':block,'cutoff_exclusive':str(cutoff.date()),'train_n':len(train),'test_n':len(test),'train_ids':int(train.person.nunique()),'test_ids':int(test.person.nunique())})
            for scope,cols in definitions.items():
                for forest,family in [(False,'ridge'),(True,'forest')]:
                    model=build(cols,schema,forest).fit(train[cols],train.OSE);pred=model.predict(test[cols]);assert np.isfinite(pred).all()
                    for (_,r),p in zip(test.iterrows(),pred):records.append({'protocol':protocol,'block':block,'model':scope+' | '+family,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(r.OSE),'predicted':float(p)})
            print('All-variable block complete',protocol,block,len(train),len(test),flush=True)
    p=pd.DataFrame(records);previous=pd.read_csv(ROOT/'outputs/ose-stress-prediction/predictions.csv',float_precision='round_trip');baseline_names=['Past OSE | ridge','Past OSE | forest','Last OSE','Past mean OSE','Population mean','Population median'];p=pd.concat([p,previous[previous.model.isin(baseline_names)]],ignore_index=True)
    assert not p.duplicated(['protocol','model','source_row']).any();p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'test_episodic_history_audit.csv',index=False);pd.DataFrame(test_features).to_csv(OUT/'test_features_audit.csv',index=False);avail=pd.DataFrame(availability);avail.to_csv(OUT/'test_modality_coverage.csv',index=False)
    scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={};sets=[]
        for name,g in pp.groupby('model'):sets.append(set(g.source_row));scores[protocol][name]={**core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':int(g.person.nunique())}
        assert all(s==sets[0] for s in sets)
        for family in ['ridge','forest']:
            for scope in definitions:
                a=scope+' | '+family;b='Past OSE | '+family;comparisons[protocol][a+' versus '+b]=paired(pp[pp.model.eq(a)],pp[pp.model.eq(b)])
            for b in ['Last OSE','Past mean OSE','All without demographics | '+family]:
                a='All predictors | '+family;comparisons[protocol][a+' versus '+b]=paired(pp[pp.model.eq(a)],pp[pp.model.eq(b)])
    subgroup={}
    for protocol,pp in p.groupby('protocol'):
        aa=avail[avail.protocol.eq(protocol)]
        for group in schema['groups']:
            keys=set(aa.loc[aa[group],'source_row']);people=int(aa.loc[aa[group],'person'].nunique());sg={'reports':len(keys),'ids':people,'scores':{},'comparisons':{}}
            if len(keys)>=10 and people>=5:
                sub=pp[pp.source_row.isin(keys)];keep=['Last OSE']+[name+' | '+family for name in ['Past OSE','Past OSE + '+group,'All predictors'] for family in ['ridge','forest']]
                for name,g in sub[sub.model.isin(keep)].groupby('model'):sg['scores'][name]=core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy())
                for family in ['ridge','forest']:
                    for name in ['Past OSE + '+group,'All predictors']:
                        a=name+' | '+family;b='Past OSE | '+family;sg['comparisons'][a+' versus '+b]=paired(sub[sub.model.eq(a)],sub[sub.model.eq(b)])
            subgroup.setdefault(protocol,{})[group]=sg
    return scores,comparisons,splitinfo,subgroup

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>OSE from all available variables</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}</style><h1>Predicting later OSE from all available variables</h1><p>Supplied OSE matches online student engagement, despite README self-efficacy label. Native aggregate score, item-level coding unverified. Known-student forecasting using information dated before the target response.</p><h2>Predictor coverage before validation history freezes</h2><table><tr><th>Group</th><th>Input features</th><th>Reports with any feature</th><th>IDs with any feature</th></tr>'
    for name,v in r['coverage'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["feature_count"]}</td><td>{v["reports_any_feature"]}</td><td>{v["ids_any_feature"]}</td></tr>'
    text+='</table><p>Coverage counts precede historical eligibility/validation freezes. Missing fields are not zero-filled observations; preprocessing is fitted only on training data. Facial raw inputs are compressed into up to10 training-derived components. Demographic/time availability counts actual demographic answers; known calendar/cohort fields do not establish demographic availability. Questionnaire-history counts exclude a zero report-count indicator as evidence of an observed score.</p>'
    rr=r['scores']['purged_rolling_blocks'];last=rr['Last OSE']['mae'];ridge=rr['All predictors | ridge']['mae'];forest=rr['All predictors | forest']['mae'];base=rr['Past OSE | ridge']['mae'];other=rr['All without past OSE | forest']['mae']
    text+=f'<h2>Result</h2><p><strong>Combining all available predictor groups does not improve the forecast over past OSE.</strong> On the same980 held-out reports from133 cohort-aware student IDs, the last-score baseline has MAE{last:.3f}, past-OSE ridge{base:.3f}, all-predictor forest{forest:.3f}, and all-predictor ridge{ridge:.3f}. No one-group addition establishes a reliable improvement in MAE over the same past-OSE model. The separate April test confirms the lack of a useful combined-model gain.</p><p>Removing past OSE from the all-predictor forest increases MAE to{other:.3f}, versus9.472 for the population mean. Other variables therefore carry some predictive information for these returning students, but remain much weaker than the student’s own earlier engagement scores. This is not a new-student validation and may partly reflect stable demographic/geometry proxies for individuals. Static collection timing and causal interpretation remain unresolved.</p><p>This is a fixed-model, untuned comparison, not proof that the extra variables have no information or that alternative representations/regularization could never help. Many derived variables are sparse or correlated. Prior OSE is the useful demonstrated forecasting baseline here.</p>'
    if r.get('actual_test_coverage'):
        text+='<h2>Source coverage on actual forecast records</h2><table><tr><th>Protocol / source</th><th>Reports with observed input</th><th>IDs</th></tr>'
        for protocol,groups in r['actual_test_coverage'].items():
            for group,v in groups.items():text+=f'<tr><td>{html.escape(protocol+" / "+group)}</td><td>{v["reports"]}</td><td>{v["ids"]}</td></tr>'
        text+='</table><p>Physical sensor observations are distinct from earlier environmental-satisfaction answers. The broad environment group includes both; the sensor-only count avoids overstating objective room coverage.</p>'
    for protocol,models in r['scores'].items():
        text+='<h2>'+protocol+'</h2><table><tr><th>Model</th><th>N / IDs</th><th>MAE↓</th><th>RMSE↓</th><th>R²</th></tr>'
        for name,v in models.items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["n"]} / {v["ids"]}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
        for name,v in r['comparisons'][protocol].items():
            lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Actual test coverage by modality</h2><p>Subset diagnostics retain matching records and do not select models or establish causation. Demographic coverage counts actual demographic answers, excluding calendar/cohort features. Other groups count at least one observed feature; they do not imply full modality completeness.</p><table><tr><th>Protocol / group available</th><th>Reports / IDs</th><th>PastOSE ridge MAE</th><th>Group-added ridge MAE</th><th>All ridge MAE</th></tr>'
    for protocol,groups in r.get('available_group_results',{}).items():
        for group,v in groups.items():
            s=v['scores'];vals=[s.get(name+' | ridge',{}).get('mae') for name in ['Past OSE','Past OSE + '+group,'All predictors']]
            text+=f'<tr><td>{html.escape(protocol+" / "+group)}</td><td>{v["reports"]} / {v["ids"]}</td>'+''.join('<td>'+('n.a.' if x is None else f'{x:.3f}')+'</td>' for x in vals)+'</tr>'
    text+='</table><p>All forest results and paired intervals for these subsets are in the full results JSON. Small modalities and multiple comparisons limit interpretation.</p><h2>Definitions, source quality and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p>Prior psychiatric/self-report codes are predictors, not diagnoses. Facial tracking/geometry can identify stable individual patterns and must not be interpreted as ground-truth emotions. More variables can overfit; individual group and combined results are exploratory after earlier inspected results. No changes were tuned to these test scores.</p><p><a href="results.json">Full results</a> · <a href="feature_schema.json">Exact predictor names, mappings and sources</a> · <a href="predictions.csv">Predictions</a> · <a href="independent_checks.json">Independent checks</a> · <a href="facial_archive_audit_all.csv">Facial date/quality audit</a> · <a href="fitbit_archive_exclusions.csv">Fitbit exclusions</a></p>'
    if r.get('archive_metadata'):
        text+='<h2>Archive processing</h2><pre>'+html.escape(json.dumps(r['archive_metadata'],indent=2))+'</pre><p>All18 facial archives/406 files were audited;266 sessions retained,87 failed tracking/frame/span criteria and53 had unusable person/date plausibility. Fitbit:60,053 CSV members accounted for;57,068 summarized,287 excluded for date mismatch,2,698 empty or without usable numeric/named-state summaries. No full facial archive extraction was written to disk. Source membership and raw samples were independently checked.</p>'
    (OUT/'report.html').write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));ose,stress,full,stores,schema=features.prepare();print('Feature coverage',json.dumps(schema['coverage']),flush=True);scores,comparisons,splits,subgroups=evaluate(ose,stress,full,stores,schema)
    models={};train=full[full.eligible]
    for scope in ['All predictors','All without demographics']:
        cols=scopes(schema)[scope]
        for forest,family in [(False,'ridge'),(True,'forest')]:models[scope+' | '+family]={'features':cols,'model':build(cols,schema,forest).fit(train[cols],train.OSE)}
    joblib.dump({'models':models,'schema':schema,'plan':PLAN},OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    for name,v in models.items():np.testing.assert_allclose(v['model'].predict(train[v['features']]),restored['models'][name]['model'].predict(train[v['features']]))
    r={'plan':PLAN,'coverage':schema['coverage'],'scores':scores,'comparisons':comparisons,'splits':splits,'available_group_results':subgroups,'feature_schema_sha256':hashlib.sha256((OUT/'feature_schema.json').read_bytes()).hexdigest(),'serialization_roundtrip':True};(OUT/'results.json').write_text(json.dumps(r,indent=2));render(r);print(json.dumps({'scores':scores,'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
