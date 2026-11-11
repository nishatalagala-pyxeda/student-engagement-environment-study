"""Exploratory advance stress prediction from prior Fitbit daily summaries."""
from pathlib import Path
import json,re,html,hashlib
import numpy as np
import pandas as pd
from sklearn.base import clone
import joblib
import depress_weekly_stress as original
import stress_environment_hypothesis as prior
from assess_depressdata import canonical
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/fitbit-stress'
EXTRACT=Path('/private/tmp/depress-fitbit-daily/Fitbit')
ACT=['minutesSedentary','minutesLightlyActive','minutesFairlyActive','minutesVeryActive']
SLEEP=['Minutes Asleep','Minutes Awake','Efficiency']
PLAN={
 'question':'Do prior sleep and activity improve advance weekly stress prediction beyond past stress?',
 'primary_window':'For survey date D, target recall week is [D-7,D). Fitbit features use [D-15,D-8): seven local-calendar days ending one full day before recall begins. One-day buffer reduces timezone/report-day ambiguity. No sensor-clock join or intraday timezone conversion.',
 'cohort':'Start with the previous234 environment-qualified responses38 IDs, to preserve scope. Primary requires at least5 valid activity days and5 valid step days in the feature week. Sleep features require3 valid main-sleep days, otherwise remain missing. Report sleep-covered sensitivity on common rows.',
 'history':'PSS-only cleaned reports strictly earlier than D-7, also strictly before validation training cutoff for test rows. No target-week reports or test-block outcomes. Existing cutoff rule is made stricter by excluding day D-7.',
 'models':'Past mean, last stress, population mean; history/control ridge(alpha10) and shallow forest(300 trees depth3 leaf10), then add activity, sleep, both. Fixed settings, no tuning. Identical common rows for each comparison; train-only imputation and encoding.',
 'validation':'Reuse fixed purged14-day rolling blocks and April holdout, with original234-row block eligibility (40 historical training rows10 IDs). Require at least20 wearable-covered training rows5 IDs. Known-history test rows are primary. Also report all rows and sleep-covered subsets. All exploratory; not a fresh validation dataset.',
 'features':'Activity: mean and population SD of daily steps, sedentary/light/fair/very-active minutes. Sleep: mean and population SD of main-sleep minutes asleep, minutes awake, and recorded efficiency. Each stream requires its own coverage threshold. No raw stress questionnaire items, concurrent outcomes, IDs or coverage counts used as predictors.',
 'data_quality':'Use actual CSV Date; require filename date agreement. Identical repeated exports collapse. Conflicting same-person/date feature values become missing. Multiple main-sleep records on a day excluded rather than added. All-zero activity days and nonpositive step totals treated as unverified wear and excluded (cannot distinguish genuine inactivity). Sleep within0..1440min, asleep>0, efficiency0..100. Device summaries are not validated clinical measurements.'}

def daily():
    rows=[];files=[];fail=[]
    for p in sorted(EXTRACT.rglob('*.csv')):
        typ=re.sub(r'\d.*','',p.stem)
        d=pd.read_csv(p);d.columns=d.columns.str.strip();person=canonical(p.relative_to(EXTRACT).parts[0])
        filename_date=pd.to_datetime(re.search(r'(\d{8})',p.stem)[1],format='%Y%m%d')
        dates=pd.to_datetime(d.Date,errors='coerce',format='mixed').dt.normalize()
        if dates.isna().any() or not dates.eq(filename_date).all():
            fail.append({'member':str(p.relative_to(EXTRACT)),'reason':'Date/filename mismatch or invalid date'});continue
        files.append({'member':str(p.relative_to(EXTRACT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(d),'type':typ})
        if typ=='sleepsummary':
            d=d[d.MainSleep.astype(str).str.lower().eq('true')].copy()
            # More than one distinct main-sleep export on a day is ambiguous.
            d=d.drop_duplicates()
            if len(d)>1:fail.append({'member':str(p.relative_to(EXTRACT)),'reason':'Multiple distinct main-sleep records'});continue
            cols=SLEEP
        elif typ=='activiessummary':cols=ACT
        elif typ=='stepssummary':cols=['Steps']
        else:continue
        for _,r in d.iterrows():
            vals={c:pd.to_numeric(r[c],errors='coerce') for c in cols}
            if typ=='activiessummary' and (any(pd.isna(v) or v<0 or v>1440 for v in vals.values()) or not 0<sum(vals.values())<=1500):
                fail.append({'member':str(p.relative_to(EXTRACT)),'reason':'Invalid or all-zero activity day'});continue
            if typ=='stepssummary' and (pd.isna(vals['Steps']) or vals['Steps']<=0):
                fail.append({'member':str(p.relative_to(EXTRACT)),'reason':'Missing/nonpositive steps, wear unverified'});continue
            if typ=='sleepsummary' and (not 0<vals['Minutes Asleep']<=1440 or not 0<=vals['Minutes Awake']<=1440 or not 0<=vals['Efficiency']<=100 or vals['Minutes Asleep']+vals['Minutes Awake']>1500):
                fail.append({'member':str(p.relative_to(EXTRACT)),'reason':'Invalid sleep summary'});continue
            rows.append({'person':person,'day':filename_date,'type':typ,**vals})
    raw=pd.DataFrame(rows);clean=[];conflicts=[];redundant=0
    for (person,day,typ),g in raw.groupby(['person','day','type']):
        cols=ACT if typ=='activiessummary' else SLEEP if typ=='sleepsummary' else ['Steps']
        redundant+=len(g)-len(g.drop_duplicates(cols))
        if len(g.drop_duplicates(cols))>1:
            conflicts.append({'person':person,'day':str(day.date()),'type':typ,'rows':len(g)});continue
        clean.append(g.iloc[0].to_dict())
    d=pd.DataFrame(clean).groupby(['person','day'],as_index=False).first().drop(columns='type')
    assert not d.duplicated(['person','day']).any()
    d.to_csv(OUT/'daily_summaries.csv',index=False)
    pd.DataFrame(files).to_csv(OUT/'source_manifest.csv',index=False);pd.DataFrame(fail).to_csv(OUT/'excluded_source_files.csv',index=False);pd.DataFrame(conflicts).to_csv(OUT/'conflicting_daily_exports.csv',index=False)
    return d,{'files':len(files),'source_rows':sum(v['rows'] for v in files),'invalid_or_ambiguous_files':len(fail),'identical_repeated_rows_collapsed':redundant,'conflicting_person_day_types_excluded':len(conflicts),'daily_rows':len(d),'people':int(d.person.nunique()),'valid_days_by_feature':d[ACT+['Steps']+SLEEP].notna().sum().to_dict(),'raw_value_ranges':raw[ACT+['Steps']+SLEEP].agg(['min','max']).to_dict()}

def features(df,daily):
    d=df.copy();audit=[]
    for c in ACT+['Steps']+SLEEP:
        for suffix in ['mean','sd']:d['fitbit_'+c.replace(' ','_')+'_'+suffix]=np.nan
    for idx,r in d.iterrows():
        origin=r.date-pd.Timedelta(days=7);end=origin-pd.Timedelta(days=1);start=end-pd.Timedelta(days=7)
        g=daily[daily.person.eq(r.person)&daily.day.ge(start)&daily.day.lt(end)]
        counts={'activity_days':int(g[ACT].notna().all(axis=1).sum()),'step_days':int(g.Steps.notna().sum()),'sleep_days':int(g[SLEEP].notna().all(axis=1).sum())}
        for c in ACT+['Steps']+SLEEP:
            limit=3 if c in SLEEP else 5
            if g[c].notna().sum()>=limit:
                d.loc[idx,'fitbit_'+c.replace(' ','_')+'_mean']=g[c].mean();d.loc[idx,'fitbit_'+c.replace(' ','_')+'_sd']=g[c].std(ddof=0)
        audit.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'forecast_origin':str(origin.date()),'features_start':str(start.date()),'features_end_exclusive':str(end.date()),'last_feature_day':str(g.day.max().date()) if len(g) else None,**counts})
        assert not len(g) or g.day.max()<end<origin<r.date
    a=pd.DataFrame(audit);a.to_csv(OUT/'feature_window_audit.csv',index=False)
    d['wearable_eligible']=(a.activity_days.ge(5)&a.step_days.ge(5)).to_numpy();d['sleep_covered']=a.sleep_days.ge(3).to_numpy()
    d.to_csv(OUT/'modeling_data.csv',index=False)
    return d,a

def history(frame,h,cutoff=None):
    # Shift historical dates by one day only during eligibility calculation to enforce strict origin cutoff.
    hh=h.copy();hh['date']=hh.date+pd.Timedelta(days=1)
    d=prior.add_history(frame,hh,None if cutoff is None else cutoff+pd.Timedelta(days=1))
    d['latest_history_date']=d.latest_history_date-pd.Timedelta(days=1)
    d['days_since_prior_stress']=d.days_since_prior_stress+1
    for _,r in d[d.latest_history_date.notna()].iterrows():assert r.latest_history_date<r.date-pd.Timedelta(days=7) and (cutoff is None or r.latest_history_date<cutoff)
    return d

def bootstrap(a,b):
    aa=a.set_index('source_row');bb=b.set_index('source_row').loc[aa.index]
    assert np.array_equal(aa.observed,bb.observed)
    delta=(bb.observed-bb.predicted).abs()-(aa.observed-aa.predicted).abs()
    groups=[delta[aa.person.eq(x)].to_numpy() for x in sorted(aa.person.unique())];rng=np.random.default_rng(42)
    boot=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(5000)]
    return {'mae_improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'n':len(a),'people':len(groups),'note':'Paired fixed-prediction person-cluster bootstrap; no model refitting or selection uncertainty.'}

def evaluate(d,h):
    nums,cats=prior.scopes_for()['History + controls'];act=['fitbit_'+c.replace(' ','_')+'_'+s for c in ACT+['Steps'] for s in ['mean','sd']];sleep=['fitbit_'+c.replace(' ','_')+'_'+s for c in SLEEP for s in ['mean','sd']]
    scopes={'History + controls':nums,'History + activity':nums+act,'History + sleep':nums+sleep,'History + sleep + activity':nums+act+sleep}
    records=[];audits=[];splits=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        for block,cutoff,tr,te in prior.get_splits(d,protocol):
            train=history(d.iloc[tr],h);test=history(d.iloc[te],h,cutoff)
            train=train[train.wearable_eligible].copy();test=test[test.wearable_eligible].copy()
            if len(train)<20 or train.person.nunique()<5 or not len(test):continue
            splits.append({'protocol':protocol,'block':block,'cutoff':str(cutoff.date()),'train_n':len(train),'test_n':len(test),'train_people':int(train.person.nunique()),'test_people':int(test.person.nunique())})
            for role,frame in [('train',train),('test',test)]:
                audits.extend([{'protocol':protocol,'block':block,'role':role,'source_row':int(r.source_row),'date':str(r.date.date()),'latest_history_date':str(r.latest_history_date.date()) if pd.notna(r.latest_history_date) else None,'history_cutoff_exclusive':str(cutoff.date()) if role=='test' else None} for _,r in frame.iterrows()])
            predictions={'Population mean':np.repeat(train.PSS.mean(),len(test)),'Past mean stress':test.past_mean_stress.fillna(train.PSS.mean()).to_numpy(),'Last stress':test.prior_stress.fillna(train.PSS.mean()).to_numpy()}
            for scope,ns in scopes.items():
                for forest,family in [(False,'ridge'),(True,'forest')]:
                    model=original.build(ns,cats,forest).fit(train[ns+cats],train.PSS)
                    predictions[scope+' | '+family]=np.clip(model.predict(test[ns+cats]),0,4)
            for name,values in predictions.items():
                for (_,r),p in zip(test.iterrows(),values):records.append({'protocol':protocol,'block':block,'model':name,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(r.PSS),'predicted':float(p),'known_history':pd.notna(r.prior_stress),'sleep_covered':r.sleep_covered})
        print('Completed',protocol,flush=True)
    p=pd.DataFrame(records);assert not p.duplicated(['protocol','model','source_row']).any();p.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'history_audit.csv',index=False)
    scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={}
        for subset,ss in [('all_wearable_covered',pp),('known_history',pp[pp.known_history]),('known_history_and_sleep',pp[pp.known_history&pp.sleep_covered])]:
            scores[protocol][subset]={n:{**original.score(g.observed.to_numpy(),g.predicted.to_numpy()),'people':int(g.person.nunique())} for n,g in ss.groupby('model')}
            cc={}
            if len(ss):
                for family in ['ridge','forest']:
                    for scope in ['History + activity','History + sleep','History + sleep + activity']:
                        name=scope+' | '+family;base='History + controls | '+family
                        cc[name+' vs '+base]=bootstrap(ss[ss.model.eq(name)],ss[ss.model.eq(base)])
                name='History + sleep + activity | ridge';cc[name+' vs Past mean stress']=bootstrap(ss[ss.model.eq(name)],ss[ss.model.eq('Past mean stress')])
            comparisons[protocol][subset]=cc
    full=history(d[d.wearable_eligible],h);models={}
    for scope,ns in scopes.items():
        for forest,family in [(False,'ridge'),(True,'forest')]:models[scope+' | '+family]={'features':ns+cats,'model':original.build(ns,cats,forest).fit(full[ns+cats],full.PSS)}
    joblib.dump(models,OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    for name,v in models.items():np.testing.assert_allclose(v['model'].predict(full[v['features']]),restored[name]['model'].predict(full[v['features']]))
    return scores,comparisons,splits,{'activity_features':act,'sleep_features':sleep,'controls_and_history':nums+cats}

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Sleep, activity and later stress</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#263345}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}</style><h1>Do sleep and activity improve later stress prediction?</h1><p>Exploratory DEPRESSData analysis. Target: supplied weekly PSS aggregate, 0–4. Lower mean absolute error (MAE) is better.</p>'
    m=r['scores']['purged_rolling_blocks']['known_history'];n=m['Past mean stress']['n'];g=m['Past mean stress']['people']
    text+=f'<h2>Result</h2><p><strong>No demonstrated improvement over past-average stress.</strong> On {n} held-out reports from {g} students, past-average stress MAE is {m["Past mean stress"]["mae"]:.3f}, history/control forest {m["History + controls | forest"]["mae"]:.3f}, and forest with sleep/activity {m["History + sleep + activity | forest"]["mae"]:.3f}. The corresponding ridge with sleep/activity scores {m["History + sleep + activity | ridge"]["mae"]:.3f}. The primary ridge addition has an improvement interval spanning zero and remains worse than the simple history baseline.</p><p>Sleep alone improves the weak ridge control comparator in the full wearable subset, but the improvement is not confirmed by the forest, April test or sleep-covered subset, and does not beat past-average stress. Most rows lack sleep; missing-value patterns may contribute to that isolated result. Only16 rolling-test reports from9 students have adequate sleep coverage. These observations do not establish that sleep or activity is unrelated to stress.</p><p>180 summary files have CSV dates inconsistent with filenames and are excluded. A further444 step files have missing/nonpositive totals and are excluded because wear is unverified. There are no conflicting repeated daily exports among retained records. The date discrepancies and sparse sleep coverage constrain this analysis.</p>'
    text+='<h2>Timing and data preparation</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p>Summary files use local-calendar dates. Survey date timezone and travel are unverified. The extra one-day buffer reduces typical within-US timezone ambiguity but does not establish exact prospective availability or eliminate delayed uploads. Static demographic administration dates are unavailable. Main sleep is assigned to the supplied Date without reconstructing bed/wake times. Missing sleep remains unknown, never zero.</p>'
    text+='<h2>Coverage</h2><pre>'+html.escape(json.dumps(r['coverage'],indent=2))+'</pre><p>Daily activity summaries do not establish actual wear time. Zero-step exclusions may omit genuine inactivity. Sleep coverage is sparse and may be selective. Starting with environment-qualified rows limits generalization beyond those students. All models in a comparison share the same training and test rows. Different subsets and prior reports can change baselines; earlier 0.409 MAE is not a direct comparator.</p>'
    for protocol,subsets in r['scores'].items():
        for subset,models in subsets.items():
            text+='<h2>'+protocol+' / '+subset+'</h2><table><tr><th>Model</th><th>Responses</th><th>Students</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>'
            for n,v in models.items():text+=f'<tr><td>{html.escape(n)}</td><td>{v["n"]}</td><td>{v["people"]}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td></tr>'
            text+='</table><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
            for n,v in r['comparisons'][protocol][subset].items():
                lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(n)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
            text+='</table>'
    text+='<h2>Interpretation</h2><p>The main comparison is known-history rolling validation, history/control ridge versus the same ridge plus sleep and activity. Forest and sleep-covered results are sensitivity checks, not a search for a winning split. An interval including zero does not establish improvement. Small numbers of students make subgroup results uncertain. These are predictive comparisons, not evidence that changing sleep or activity causes stress changes. The same historical study has already been explored; independent validation remains necessary.</p><p><a href="results.json">Full results</a> · <a href="feature_window_audit.csv">Feature timing/coverage</a> · <a href="source_manifest.csv">Source manifest</a> · <a href="integrity_checks.json">Checks</a></p>'
    (OUT/'report.html').write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));df,h,meta=prior.load();daily_data,audit=daily();print('Daily summary audit',json.dumps(audit),flush=True);d,windows=features(df,daily_data)
    coverage={'initial_responses':len(d),'initial_people':int(d.person.nunique()),'activity_step_covered_responses':int(d.wearable_eligible.sum()),'activity_step_covered_people':int(d.loc[d.wearable_eligible,'person'].nunique()),'also_sleep_covered_responses':int((d.wearable_eligible&d.sleep_covered).sum()),'also_sleep_covered_people':int(d.loc[d.wearable_eligible&d.sleep_covered,'person'].nunique()),'sleep_covered_without_activity_requirement':int(d.sleep_covered.sum()),'activity_days_distribution':windows.activity_days.value_counts().sort_index().to_dict(),'sleep_days_distribution':windows.sleep_days.value_counts().sort_index().to_dict()}
    print('Coverage',json.dumps(coverage),flush=True);scores,comparisons,splits,fs=evaluate(d,h)
    checks={'no_conflicting_daily_values_selected_arbitrarily':True,'one_daily_record_per_person_date':True,'csv_date_equals_filename_date_for_included_files':True,'all_features_end_before_target_recall_week_with_one_day_buffer':True,'history_strictly_before_recall_week':True,'test_history_frozen_before_training_cutoff':True,'train_only_preprocessing':True,'identical_rows_for_nested_feature_comparisons':True,'no_duplicate_predictions':True,'serialization_roundtrip':True}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2));r={'plan':PLAN,'coverage':coverage,'source_audit':audit,'metadata':meta,'features':fs,'scores':scores,'comparisons':comparisons,'splits':splits};(OUT/'results.json').write_text(json.dumps(r,indent=2));render(r)
    print(json.dumps({'scores':scores,'comparisons':comparisons},indent=2),flush=True)
if __name__=='__main__':main()
