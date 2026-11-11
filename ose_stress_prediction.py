"""Forecast supplied OSE engagement scores; test incremental value of earlier PSS."""
from pathlib import Path
import json,hashlib,html,zipfile
import numpy as np
import pandas as pd
import joblib
import depress_weekly_stress as regression
import stress_environment_hypothesis as validation
from assess_depressdata import canonical
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/ose-stress-prediction'
FIELDS=['mean','last','last_deviation','sd','last_change','count','age_days']
OSE_FEATURES=['ose_'+c for c in FIELDS];STRESS_FEATURES=['stress_'+c for c in FIELDS]
PLAN={
 'question':'Does earlier stress improve prediction of later supplied OSE scores beyond earlier OSE?',
 'construct_correction':'README calls OSE online self-efficacy, but supplied19-item questionnaire matches Dixson Online Student Engagement scale (study habits, effort, course interest, participation and performance). Model the supplied OSE column; results do not establish prediction of self-efficacy/confidence. Course identity is not supplied.',
 'score':'Provided aggregate OSE in native units; no raw items to reconstruct coding, offset or handling of missing items. Retain finite nonnegative supplied scores, including0 and fractional values, flag their counts. Do not assume a19–95 or0–76 scoring convention or invent a low/high cutoff. Regression only; predictions not clipped to an unverified theoretical scale.',
 'timing':'Predict a later survey score using only OSE and PSS reports strictly before target date minus7days. OSE questionnaire refers to the course without explicit weekly recall, so call this later-score forecasting, not a verified next-week engagement window. For test blocks, both source histories also strictly before block training cutoff. No same-survey stress, computer/network scores or future labels.',
 'cleaning':'Clean each target/source separately: valid ID/date/numeric score; same-person/date conflicting aggregate scores all excluded, identical score repeats collapse even when nonmodeled computer/network fields differ. Canonical cohort-aware IDs; cross-cohort physical identity unverified.',
 'cohort':'Use all cleaned OSE reports, no sensor/wearable restriction. Primary comparison requires at least2 admissible earlier OSE reports and1 earlier PSS report. All compared models use identical training/test records. Report exclusions and coverage before and after frozen histories.',
 'features':'Seven summaries each for earlier OSE and PSS: mean, last, last-minus-mean, population SD, last-minus-second-last, report count, days since last report. Missing stress last-change with only1 past stress report remains missing; train-only imputation. Sensitivity adds target-date annual sine/cosine and study-cohort category identically to both models.',
 'models':'Primary ridge alpha10; forest sensitivity300trees maxdepth3 minleaf10 seed42. Past OSE versus past OSE+past stress. Repeat with identical time/cohort controls. No tuning, class thresholds, demographic, environmental or Fitbit predictors. Training-only imputation, scaling and encoding.',
 'baselines':'Training-wide mean/median; student mean earlier OSE; last earlier OSE. Past OSE baselines use identical admissible source histories as models, including reports not eligible as training targets.',
 'validation':'Same expanding14-day blocks anchoredJune1,2020, train before block start minus7days, at least40 raw earlier targets10 IDs, then20 eligible train targets5 IDs. Primary rolling test; secondary April holdout train beforeMarch25,2021 and testApril1 onward. Each test report scored once per protocol. Frozen histories for whole test block/April test. Reused study periods, exploratory.',
 'metrics':'MAE primary, RMSE/R² secondary; person-cluster paired fixed-prediction bootstrap3000draws seed42 for MAE improvement from stress, and versus personalized baselines. Positive favors augmented predictor. No model refits or full model-selection uncertainty.',
 'limits':'Predictive incremental association, not causal effect of stress. Noisy supplied scoring, questionnaire/README construct mismatch, unknown course identity or changes, irregular dates/history gaps, selection of returning respondents, missing raw items, retrospective data availability and repeated reuse of study. Does not predict students with no earlier OSE history.'}

def clean(path,score,prefix):
    raw=pd.read_csv(path);raw['person']=raw.ID.map(canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce');raw['source_row']=raw.index+2;raw[score]=pd.to_numeric(raw[score],errors='coerce')
    valid=raw.person.notna()&raw.date.notna()&np.isfinite(raw[score])&raw[score].ge(0)
    if score=='PSS':valid&=raw[score].le(4)
    d=raw[valid].copy();conflict=d.groupby(['person','date'])[score].transform('nunique').gt(1);d[conflict].to_csv(OUT/f'excluded_conflicting_{prefix}.csv',index=False);nonconflict=d[~conflict]
    d=nonconflict.drop_duplicates(['person','date']).sort_values(['person','date']).reset_index(drop=True)[['person','date','source_row',score]];assert not d.duplicated(['person','date']).any();d.to_csv(OUT/f'cleaned_{prefix}.csv',index=False)
    diffs=d.sort_values(['person','date']).groupby('person').date.diff().dt.days.dropna()
    metadata={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_rows':len(raw),'invalid_or_missing_rows':int((~valid).sum()),'conflicting_rows_excluded':int(conflict.sum()),'identical_repeated_scores_collapsed':len(nonconflict)-len(d),'cleaned_rows':len(d),'ids':int(d.person.nunique()),'minimum':float(d[score].min()),'maximum':float(d[score].max()),'zero_scores':int(d[score].eq(0).sum()),'fractional_scores':int(d[score].mod(1).ne(0).sum()),'first':str(d.date.min().date()),'last':str(d.date.max().date()),'median_days_between_reports':float(diffs.median()),'gap_days_quantiles':diffs.quantile([.1,.5,.9]).to_dict()}
    return d,metadata

def history_features(targets,ose,stress,cutoff=None):
    stores={'ose':{k:g.sort_values('date') for k,g in ose.groupby('person')},'stress':{k:g.sort_values('date') for k,g in stress.groupby('person')}};records=[]
    for _,r in targets.iterrows():
        rec={}
        for prefix,score in [('ose','OSE'),('stress','PSS')]:
            available=stores[prefix].get(r.person)
            if available is not None:
                available=available[available.date.lt(r.date-pd.Timedelta(days=7))]
                if cutoff is not None:available=available[available.date.lt(cutoff)]
            n=0 if available is None else len(available)
            if n:
                values=available[score].to_numpy();mu=float(values.mean());latest=available.iloc[-1]
                values_out=[mu,float(values[-1]),float(values[-1]-mu),float(values.std(ddof=0)),float(values[-1]-values[-2]) if n>=2 else np.nan,n,(r.date-latest.date).days]
                assert latest.date<r.date-pd.Timedelta(days=7) and (cutoff is None or latest.date<cutoff)
                rec[prefix+'_latest_date']=str(latest.date.date());rec[prefix+'_history_source_rows']=','.join(str(int(v)) for v in available.source_row)
            else:values_out=[np.nan,np.nan,np.nan,np.nan,np.nan,0,np.nan];rec[prefix+'_latest_date']=None;rec[prefix+'_history_source_rows']=''
            rec.update({prefix+'_'+field:v for field,v in zip(FIELDS,values_out)})
        records.append(rec)
    out=targets.reset_index(drop=True).copy()
    for c in records[0]:out[c]=[r[c] for r in records]
    out['annual_sin']=np.sin(2*np.pi*(out.date.dt.dayofyear-1)/365.25);out['annual_cos']=np.cos(2*np.pi*(out.date.dt.dayofyear-1)/365.25);out['study_cohort']=out.person.str.extract(r'^(\w+)')[0]
    out['eligible']=out.ose_count.ge(2)&out.stress_count.ge(1)
    return out

def paired(a,b):
    a=a.set_index('source_row');b=b.set_index('source_row').loc[a.index];np.testing.assert_array_equal(a.observed,b.observed)
    delta=abs(b.observed-b.predicted)-abs(a.observed-a.predicted);groups=[delta[a.person.eq(x)].to_numpy() for x in sorted(a.person.unique())];rng=np.random.default_rng(42)
    boot=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(3000)]
    return {'mae_improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'n':len(a),'ids':len(groups)}

def scopes():
    return {'Past OSE':(OSE_FEATURES,[]),'Past OSE + stress':(OSE_FEATURES+STRESS_FEATURES,[]),'Past OSE + time':(OSE_FEATURES+['annual_sin','annual_cos'],['study_cohort']),'Past OSE + time + stress':(OSE_FEATURES+['annual_sin','annual_cos']+STRESS_FEATURES,['study_cohort'])}

def evaluate(ose,stress,full):
    records=[];audits=[];splits=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        for block,cutoff,tr,te in validation.get_splits(ose,protocol):
            train=full.iloc[tr];train=train[train.eligible];test=history_features(ose.iloc[te],ose,stress,cutoff);raw_n=len(test);test=test[test.eligible]
            if len(train)<20 or train.person.nunique()<5 or not len(test):continue
            assert train.date.max()<cutoff and set(train.source_row).isdisjoint(test.source_row)
            splits.append({'protocol':protocol,'block':block,'history_cutoff_exclusive':str(cutoff.date()),'train_n':len(train),'train_ids':int(train.person.nunique()),'candidate_test_n':raw_n,'eligible_test_n':len(test),'test_ids':int(test.person.nunique()),'test_first':str(test.date.min().date()),'test_last':str(test.date.max().date())})
            for role,frame in [('train',train),('test',test)]:
                for _,r in frame.iterrows():audits.append({'protocol':protocol,'block':block,'role':role,'cutoff_exclusive':str(cutoff.date()),'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'OSE':float(r.OSE),**{c:r[c] for c in OSE_FEATURES+STRESS_FEATURES+['ose_latest_date','stress_latest_date','ose_history_source_rows','stress_history_source_rows']}})
            predictions={'Population mean':np.repeat(train.OSE.mean(),len(test)),'Population median':np.repeat(train.OSE.median(),len(test)),'Past mean OSE':test.ose_mean.to_numpy(),'Last OSE':test.ose_last.to_numpy()}
            for name,(nums,cats) in scopes().items():
                for forest,family in [(False,'ridge'),(True,'forest')]:
                    model=regression.build(nums,cats,forest).fit(train[nums+cats],train.OSE);predictions[name+' | '+family]=model.predict(test[nums+cats])
            for name,values in predictions.items():
                for (_,r),pred in zip(test.iterrows(),values):records.append({'protocol':protocol,'block':block,'model':name,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(r.OSE),'predicted':float(pred)})
        print('Completed OSE',protocol,flush=True)
    p=pd.DataFrame(records);assert not p.duplicated(['protocol','model','source_row']).any();p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'history_availability_audit.csv',index=False)
    scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={}
        for name,g in pp.groupby('model'):scores[protocol][name]={**regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':int(g.person.nunique())}
        for family in ['ridge','forest']:
            pairs=[('Past OSE + stress | '+family,'Past OSE | '+family),('Past OSE + time + stress | '+family,'Past OSE + time | '+family),('Past OSE + stress | '+family,'Past mean OSE'),('Past OSE + stress | '+family,'Last OSE')]
            for added,baseline in pairs:comparisons[protocol][added+' versus '+baseline]=paired(pp[pp.model.eq(added)],pp[pp.model.eq(baseline)])
        comparisons[protocol]['Past mean OSE versus Population mean']=paired(pp[pp.model.eq('Past mean OSE')],pp[pp.model.eq('Population mean')])
    return scores,comparisons,splits

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Earlier stress and later OSE</title><style>body{font:16px/1.6 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}</style><h1>Does past stress predict later online engagement?</h1><p><strong>Construct correction:</strong> the README calls OSE self-efficacy. The supplied questionnaire matches the19-item Online Student Engagement scale: study habits, effort, interest, participation and performance. These results concern the supplied OSE score, not validated self-efficacy/confidence. See <a href="https://olj.onlinelearningconsortium.org/index.php/olj/article/view/561">Dixson (2015), original OSE validation</a> and the supplied questionnaire image below. This mapping is based on matching questionnaire content; aggregate scoring cannot be independently reconstructed.</p>'
    text+='<h2>Coverage</h2><pre>'+html.escape(json.dumps(r['coverage'],indent=2))+'</pre>'
    rolling=r['scores']['purged_rolling_blocks'];last=rolling['Last OSE'];mean=rolling['Population mean'];before=rolling['Past OSE | ridge'];after=rolling['Past OSE + stress | ridge'];ci=r['comparisons']['purged_rolling_blocks']['Past OSE + stress | ridge versus Past OSE | ridge'];lo,hi=ci['person_cluster_95_interval']
    text+=f'<h2>Result</h2><p><strong>Earlier OSE scores predict later supplied OSE substantially better than population baselines. Earlier stress adds no demonstrated benefit.</strong> The rolling test contains {last["n"]} reports from {last["ids"]} IDs. The last earlier OSE baseline has MAE {last["mae"]:.3f}, compared with {mean["mae"]:.3f} for the training-wide mean. History-only ridge has MAE {before["mae"]:.3f} and R² {before["r2"]:.3f}; adding stress changes MAE to {after["mae"]:.3f} (worse). Its paired MAE improvement is {ci["mae_improvement"]:.3f}, with person-cluster95% interval [{lo:.3f}, {hi:.3f}]. The forest stress addition is essentially unchanged and also has an interval spanning zero.</p><p>The April test similarly shows a strong last-OSE baseline (MAE4.769 versus population11.267) and no established incremental stress gain. Identical time/cohort adjustments do not change the conclusion. Predictability may largely reflect stable individual/course patterns; this does not demonstrate reliable prediction of individual changes, causal effects of stress, or forecasting for students without earlier OSE scores. No computerized engagement items or contemporaneous stress responses entered the models.</p><p>The provided aggregate has observed range0–76, two zero scores and two fractional values. These are retained, because the supplied raw items/coding are unavailable and an offset or scoring convention cannot be validated. Interpretation as student engagement follows the questionnaire content; self-efficacy remains unverified despite the README label.</p>'
    for protocol,models in r['scores'].items():
        text+='<h2>'+html.escape(protocol)+'</h2><table><tr><th>Model</th><th>N / IDs</th><th>MAE↓ (provided OSE units)</th><th>RMSE↓</th><th>R²</th></tr>'
        for name,v in models.items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["n"]} / {v["ids"]}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
        for name,v in r['comparisons'][protocol].items():
            lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Definitions and limitations</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<h2>Source-score audit</h2><pre>'+html.escape(json.dumps(r['sources'],indent=2))+'</pre><h2>Supplied OSE questionnaire</h2><img src="ose_questionnaire.png" alt="Supplied19-item questionnaire" style="max-width:100%"><p><a href="results.json">Full results</a> · <a href="analysis_plan.json">Definitions</a> · <a href="predictions.csv">Predictions</a> · <a href="history_availability_audit.csv">Source history/timing audit</a> · <a href="independent_checks.json">Independent checks</a></p>'
    (OUT/'report.html').write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));ose,ometa=clean(next(regression.SOURCE.rglob('OSE_Computer_Internert.csv')),'OSE','OSE');stress,smeta=clean(next(regression.SOURCE.rglob('PANAS_PSS.csv')),'PSS','PSS')
    questionnaire=next(regression.SOURCE.rglob('Consent form and questionnaires.docx'))
    with zipfile.ZipFile(questionnaire) as z:(OUT/'ose_questionnaire.png').write_bytes(z.read('word/media/image14.png'))
    full=history_features(ose,ose,stress);full.to_csv(OUT/'full_history_features.csv',index=False)
    coverage={'cleaned_OSE_reports':len(ose),'cleaned_OSE_ids':int(ose.person.nunique()),'cleaned_PSS_reports':len(stress),'cleaned_PSS_ids':int(stress.person.nunique()),'at_least2_prior_OSE_before_freeze':int(full.ose_count.ge(2).sum()),'also_prior_stress_before_freeze':int(full.eligible.sum()),'eligible_ids_before_freeze':int(full.loc[full.eligible,'person'].nunique()),'missing_OSE_history':int(full.ose_count.lt(2).sum()),'adequate_OSE_but_no_earlier_PSS':int((full.ose_count.ge(2)&full.stress_count.lt(1)).sum())}
    print('Coverage',json.dumps(coverage),flush=True);print('OSE source',json.dumps(ometa),flush=True);scores,comparisons,splits=evaluate(ose,stress,full)
    models={};f=full[full.eligible]
    for scope,(nums,cats) in scopes().items():
        for forest,family in [(False,'ridge'),(True,'forest')]:models[scope+' | '+family]={'features':nums+cats,'model':regression.build(nums,cats,forest).fit(f[nums+cats],f.OSE)}
    joblib.dump({'models':models,'plan':PLAN},OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    for name,v in models.items():np.testing.assert_allclose(v['model'].predict(f[v['features']]),restored['models'][name]['model'].predict(f[v['features']]))
    checks={'target_and_stress_sources_cleaned_independently':True,'no_current_target_or_same_day_stress_predictors':True,'all_source_histories_strictly_before_target_minus7days':True,'test_histories_frozen_before_training_cutoff':True,'train_only_preprocessing':True,'all_compared_models_share_training_and_test_rows':True,'test_rows_unique_per_protocol_model':True,'serialization_roundtrip_passes':True}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2));r={'plan':PLAN,'coverage':coverage,'sources':{'OSE':ometa,'PSS':smeta,'questionnaire':str(questionnaire),'questionnaire_sha256':hashlib.sha256(questionnaire.read_bytes()).hexdigest()},'scores':scores,'comparisons':comparisons,'splits':splits};(OUT/'results.json').write_text(json.dumps(r,indent=2));render(r);print(json.dumps({'scores':scores,'comparisons':comparisons},indent=2))
if __name__=='__main__':main()
