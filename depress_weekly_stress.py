"""DEPRESSData weekly PSS regression: personal bedroom exposures, grouped/forward evaluation."""
from pathlib import Path
import os,json,hashlib,sys,platform,html
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from assess_depressdata import canonical
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Users/nishatalagala/Downloads/DEPRESSData')
OUT=ROOT/'outputs/depress-weekly-stress'
PRIMARY='Environment + personal/time | ridge'
FORECAST_START=pd.Timestamp('2021-04-01')
GAP=pd.Timedelta(days=7)
BOUNDS=(0,4)


def prepare():
    assessment=ROOT/'outputs/depressdata-assessment'
    mapping=json.loads((assessment/'sensor_id_mapping.json').read_text());mp={v['sensor_id']:v['proposed_person'] for v in mapping if v['proposed_person']}
    assert len(mp)==len(set(mp.values()))==41 and 'F135' not in mp
    overlap=pd.read_csv(assessment/'survey_environment_overlap.csv')
    qualified=overlap[overlap.table.eq('weekly')&overlap.at_least_5_days_72plus_slots].copy()
    assert len(qualified)==234 and qualified.person.nunique()==38
    df=qualified[['source_row','person','survey_date','PSS']].copy().reset_index(drop=True);df['date']=pd.to_datetime(df.survey_date);df['exposure_start']=df.date-GAP;df['exposure_end_exclusive']=df.date
    weekly_path=next(SOURCE.rglob('PANAS_PSS.csv'));weekly=pd.read_csv(weekly_path)
    s=weekly.iloc[(df.source_row-2).astype(int)].reset_index(drop=True)
    assert np.array_equal(s.ID.map(canonical),df.person)
    np.testing.assert_allclose(s.PSS,df.PSS,atol=0,rtol=0)
    dates=pd.to_datetime(s.StartDate,format='mixed',errors='raise');assert np.array_equal(dates,df.date)
    assert df.PSS.between(*BOUNDS).all() and not df.duplicated(['person','date']).any()
    env_path=next(SOURCE.rglob('Bedroom_IEQ.csv'));e=pd.read_csv(env_path);e['_person']=e.ID.map(mp);e['_local']=pd.to_datetime(e['timestamp(America/New_York)'],format='mixed')
    parts=[]
    for _,g in e.groupby('ID',sort=False):
        t=pd.DatetimeIndex(g._local).tz_localize('America/New_York',ambiguous='infer',nonexistent='NaT').tz_convert('UTC');assert t.notna().all();parts.append(pd.Series(t,index=g.index))
    e['_utc']=pd.concat(parts).sort_index();assert not e.duplicated(['ID','_utc']).any()
    e=e[e._person.notna()].copy();e['_date']=e._local.dt.normalize()
    temp=next(c for c in e if c.lower().startswith('temp('))
    signals={'temperature_C':temp,'humidity':'humid','co2':'co2','voc':'voc','pm25':'pm25','light':'light','noise':'noise'}
    for field,col in signals.items():e[field]=((e[col]-32)*5/9) if field=='temperature_C' else e[col]
    assert np.isfinite(e[list(signals)].to_numpy()).all()
    assert e.temperature_C.between(-20,60).all() and e.humidity.between(0,100).all()
    assert e[['co2','voc','pm25','light','noise']].ge(0).all().all()
    daily=e.groupby(['_person','_date'])[list(signals)].mean()
    daily['wall_clock_slots']=e.groupby(['_person','_date'])._local.nunique()
    good=daily[daily.wall_clock_slots.ge(72)].copy()
    numeric=[]
    for field in signals:
        for suffix in ['week_mean','between_day_sd']:
            name=field+'_'+suffix;df[name]=np.nan;numeric.append(name)
    audits=[]
    for i,row in df.iterrows():
        day=good.loc[row.person].copy();day=day[(day.index>=row.exposure_start)&(day.index<row.date)]
        assert 5<=len(day)<=7 and day.index.max()<row.date
        for field in signals:
            df.loc[i,field+'_week_mean']=day[field].mean();df.loc[i,field+'_between_day_sd']=day[field].std(ddof=0)
        audits.append({'source_row':int(row.source_row),'person':row.person,'survey_date':str(row.date.date()),'exposure_start':str(row.exposure_start.date()),'exposure_end_exclusive':str(row.date.date()),'good_days':len(day),'good_day_dates':','.join(day.index.strftime('%Y-%m-%d')),'good_day_wall_slots':int(day.wall_clock_slots.sum())})
    assert df[numeric].notna().all().all()
    # Static covariates are joined once per participant, without target-based filling.
    demo_path=next(SOURCE.rglob('*.xlsx'));demo=pd.read_excel(demo_path);demo['_person']=demo.ID.map(canonical);assert demo._person.is_unique
    background=demo.set_index('_person').reindex(df.person.to_numpy()).rename_axis('_person').reset_index()
    assert np.array_equal(background._person,df.person) and background.index.equals(df.index)
    birth=pd.to_numeric(background.YearBirth,errors='coerce');df['approximate_age']=df.date.dt.year-birth
    df['socioeconomic_ladder']=pd.to_numeric(background.Ladder,errors='coerce')
    cats=[]
    for name,original in [('recorded_sex','Sex'),('income_category','Income'),('household_size_category','NumPeopleLive'),('bedroom_count_category','NumBedrooms')]:
        df[name]=background[original].map(lambda v:str(v) if pd.notna(v) else np.nan);cats.append(name)
    # Retain open-ended original categories rather than pretending '5+' or '>6' are exact quantities.
    df['study_cohort']=df.person.str.extract(r'^(\w+)')[0]
    dayofyear=df.date.dt.dayofyear-1;df['annual_sin']=np.sin(2*np.pi*dayofyear/365.25);df['annual_cos']=np.cos(2*np.pi*dayofyear/365.25)
    time_num=['annual_sin','annual_cos'];time_cat=['study_cohort'];personal_num=['approximate_age','socioeconomic_ladder']
    scopes={'Time':(time_num,time_cat),'Personal/time':(personal_num+time_num,cats+time_cat),'Environment':(numeric,[]),'Environment + time':(numeric+time_num,time_cat),'Environment + personal/time':(numeric+personal_num+time_num,cats+time_cat)}
    for nums,cs in scopes.values():
        assert not set(nums+cs)&{'PSS','person','source_row','date','ID','Positive emotion','Negative emotion','CESD','STAI_st'}
        assert len(nums+cs)==len(set(nums+cs))
    metadata={'n':len(df),'people':int(df.person.nunique()),'dates':int(df.date.nunique()),'target':'Provided weekly PSS score; observed original aggregate on0..4. Four-item instrument refers to previous week. Item-level scoring/reverse coding is not available for independent validation; no clinical interpretation or cutoffs.','pss_summary':df.PSS.describe().to_dict(),'primary_model':PRIMARY,'environment_features':numeric,'scope_features':{n:nums+cs for n,(nums,cs) in scopes.items()},'people_missing_demographics':int(background.loc[background.YearBirth.isna(),'_person'].nunique()),'missing_covariates':df[personal_num+cats].isna().sum().to_dict(),'cohort_counts':df.study_cohort.value_counts().to_dict(),'environment_preparation':'Previous7 complete New York calendar days, excluding survey date. A good day has at least72 distinct15-minute wall-clock slots. Require at least5 good days; equal-weight average and population SD of good-day means for each of7 measurements. Coverage never uses outcome values. Sensor real timestamps inferred in recorded order across DST; repeated fall-back hour retained.','noise_note':'Arithmetic mean of reported noise values is descriptive; not energy-equivalent sound level. VOC/PM extremes retained rather than discarded without sensor calibration evidence.','personal_note':'Approximate age=survey year minus numeric birth year, not exact birthday age. Nonnumeric birth years/ladder remain missing. Sex/income/household size/bedrooms retain original categories, including open-ended responses. Static covariate administration date is not available. Missing demographics retained with training-only imputation.','mapping_note':'41 one-to-one lexical sensor-to-survey candidates; F135 unresolved and excluded. Author confirmation is still unavailable. Student grouping uses supplied participant/cohort IDs; cross-cohort physical identity linkage is not established.','forward_test_start':str(FORECAST_START.date()),'forward_training_rule':'Survey date strictly before2021-03-25; test date on/after2021-04-01. Seven-day gap; no test outcomes used for model or personalized baselines.','window_interpretation':'Predict reported stress for the exposure week just ended. This is not a next-week stress forecast. Survey time-of-day is absent, so the entire survey day is excluded.','dependencies':{m.__name__:m.__version__ for m in [np,pd,sklearn,scipy,joblib]},'python':sys.version,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [weekly_path,env_path,demo_path,assessment/'sensor_id_mapping.json',assessment/'survey_environment_overlap.csv']},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    pd.DataFrame(audits).to_csv(OUT/'exposure_window_audit.csv',index=False)
    return df,scopes,metadata


def build(nums,cats,forest):
    prep=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler()),nums),('categorical',make_pipeline(SimpleImputer(strategy='constant',fill_value='Unknown'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)],remainder='drop')
    model=RandomForestRegressor(n_estimators=300,max_depth=3,min_samples_leaf=10,max_features=1.0,random_state=42,n_jobs=1) if forest else Ridge(alpha=10.0)
    return make_pipeline(prep,model)


def specifications(scopes):
    models={'Mean baseline':(['annual_sin'],DummyRegressor(strategy='mean')),'Median baseline':(['annual_sin'],DummyRegressor(strategy='median'))}
    for name,(nums,cats) in scopes.items():
        for forest,family in [(False,'ridge'),(True,'forest')]:models[name+' | '+family]=(nums+cats,build(nums,cats,forest))
    return models


def make_splits(df,protocol):
    if protocol=='held_out_student':
        for person in sorted(df.person.unique()):
            train=np.flatnonzero(df.person.ne(person));test=np.flatnonzero(df.person.eq(person));assert set(df.iloc[train].person).isdisjoint(df.iloc[test].person)
            yield train,test
    elif protocol=='forward_time':
        train=np.flatnonzero(df.date.lt(FORECAST_START-GAP));test=np.flatnonzero(df.date.ge(FORECAST_START))
        assert len(train) and len(test) and df.iloc[train].date.max()+GAP<=df.iloc[test].date.min()
        # No raw-exposure intervals overlap across the train/test boundary, even within one student.
        assert df.iloc[train].exposure_end_exclusive.max()<=df.iloc[test].exposure_start.min()
        yield train,test
    else:raise ValueError(protocol)


def score(actual,predicted):
    return {'n':len(actual),'mae':float(mean_absolute_error(actual,predicted)),'rmse':float(np.sqrt(mean_squared_error(actual,predicted))),'r2':float(r2_score(actual,predicted)),'bias':float(np.mean(predicted-actual))}


def evaluate(df,models,protocol):
    scores={};records=[];folds=list(make_splits(df,protocol));y=df.PSS.to_numpy(float)
    for name,(cols,template) in models.items():
        p=np.full(len(df),np.nan);raw=p.copy()
        for train,test in folds:
            assert set(train).isdisjoint(test);model=clone(template).fit(df.iloc[train][cols],y[train]);raw[test]=model.predict(df.iloc[test][cols]);p[test]=np.clip(raw[test],*BOUNDS)
        used=np.isfinite(p);scores[name]=score(y[used],p[used]);scores[name]['clipped_predictions']=int((raw[used]!=p[used]).sum())
        for i in np.flatnonzero(used):records.append({'protocol':protocol,'model':name,'source_row':int(df.iloc[i].source_row),'person':df.iloc[i].person,'survey_date':str(df.iloc[i].date.date()),'observed':float(y[i]),'predicted':float(p[i]),'unbounded_prediction':float(raw[i])})
    # Personalized simple baselines use ONLY fold-training outcomes, never held-out outcomes.
    for name in ['Past-person mean baseline','Last past-person score baseline']:
        p=np.full(len(df),np.nan)
        for train,test in folds:
            training=df.iloc[train];fallback=training.PSS.mean()
            for i in test:
                row=df.iloc[i];past=training[training.person.eq(row.person)&training.date.lt(row.date)].sort_values('date')
                p[i]=fallback if past.empty else (past.PSS.mean() if name.startswith('Past-person mean') else past.PSS.iloc[-1])
        used=np.isfinite(p);scores[name]=score(y[used],p[used]);scores[name]['clipped_predictions']=0
        for i in np.flatnonzero(used):records.append({'protocol':protocol,'model':name,'source_row':int(df.iloc[i].source_row),'person':df.iloc[i].person,'survey_date':str(df.iloc[i].date.date()),'observed':float(y[i]),'predicted':float(p[i]),'unbounded_prediction':float(p[i])})
    return scores,pd.DataFrame(records)


def paired_intervals(predictions):
    comparisons=[(PRIMARY,'Mean baseline'),(PRIMARY,'Median baseline'),(PRIMARY,'Personal/time | ridge'),('Environment + personal/time | forest','Mean baseline'),('Environment + personal/time | forest','Personal/time | forest'),('Environment | ridge','Mean baseline'),('Environment | forest','Mean baseline')]
    results={}
    for protocol in predictions.protocol.unique():
        p=predictions[predictions.protocol.eq(protocol)];results[protocol]={}
        for model,baseline in comparisons:
            a=p[p.model.eq(model)].set_index('source_row');b=p[p.model.eq(baseline)].set_index('source_row').loc[a.index]
            change=np.abs(a.predicted-a.observed)-np.abs(b.predicted-b.observed);groups=[change[a.person.eq(g)].to_numpy() for g in sorted(a.person.unique())];rng=np.random.default_rng(42)
            draws=[np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(3000)]
            results[protocol][model+' versus '+baseline]={'mae_change':float(change.mean()),'person_cluster_95_interval':np.quantile(draws,[.025,.975]).tolist(),'clusters':len(groups),'note':'Negative favors model. Fixed held-out predictions resampled by person, no refitting. Does not capture training/model-selection uncertainty or shared time/cohort dependence.'}
    return results


def report(out):
    m=out['metadata'];ev=out['evaluation'];a=ev['held_out_student'][PRIMARY];f=ev['forward_time'][PRIMARY]
    text='''<!doctype html><html><head><meta charset="utf-8"><title>Weekly stress regression</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}</style></head><body><h1>DEPRESSData: weekly stress regression</h1><p>November10,2026. Exploratory research using the supplied aggregate PSS score.</p>'''
    text+=f'''<p>Same <strong>{m['n']} weekly responses, {m['people']} participant IDs and {m['dates']} survey dates</strong> for every model. PSS is modeled on its supplied0–4 scale, without inventing clinical labels. Its raw four-item responses and scoring history are absent. Outcomes are weekly self-reports, not diagnoses.</p><h2>Primary results</h2><p>The prespecified combined ridge model has held-out-student MAE {a['mae']:.3f}, compared with {ev['held_out_student']['Mean baseline']['mae']:.3f} for the mean baseline and {ev['held_out_student']['Median baseline']['mae']:.3f} for the median baseline. Forward-time MAE is {f['mae']:.3f}, compared with {ev['forward_time']['Mean baseline']['mae']:.3f} and {ev['forward_time']['Median baseline']['mae']:.3f}, respectively. Lower error is better. The tables and paired intervals below must be read together; comparator models were not selected by their test scores.</p>
    <h2>Features and timing</h2><p>{html.escape(m['environment_preparation'])} Temperature is converted from Fahrenheit to Celsius. There are14 environmental level/variability inputs: weekly means and between-day standard deviations for temperature, humidity, CO₂, VOCs, PM2.5, light and noise. The device's composite environmental score is excluded. {html.escape(m['noise_note'])}</p><p>Personal factors: approximate age, recorded sex, income category, household-size category, bedroom-count category and socioeconomic ladder. Time controls: annual sine/cosine and study-cohort category. {html.escape(m['personal_note'])} Cohort is a shared study-period label, not a participant identifier. No outcome questionnaire item, concurrent mental-health score, participant ID, exact-date category or future record is a predictor. Fitbit, daily diary, facial features and other questionnaires are reserved for later feature comparisons.</p><p>{html.escape(m['window_interpretation'])} This analysis estimates reported stress for the week with environmental measurements; it does not show that conditions caused stress.</p>
    <h2>Evaluation</h2><p>Leave one supplied participant ID out (38 folds) tests generalization to unobserved IDs. Forward-time evaluation trains on survey dates strictly before March25,2021 and tests April1,2021 onward, with a seven-day gap. Its {out['forward_split']['train_n']} training responses and {out['forward_split']['test_n']} test responses have no overlapping exposure windows across the split. {out['forward_split']['test_people']} IDs appear in the test; {out['forward_split']['seen_test_people']} have earlier matched training responses. Forward testing is in the Spring cohort and should not be taken as a test across every season/cohort.</p><p>All imputers, missing indicators, standardization and one-hot encoding are fitted in each training fold. Fixed ridge alpha10; fixed forest300 trees, depth3, minimum leaf10, random seed42, one worker. No tuning search. All predictions clipped to0–4 before scoring. Final full-data fits are exported only after held-out evaluation. Mean/median baselines are fitted separately per fold. Personalized baselines use only earlier outcomes in that fold's training set; unseen students fall back to the training mean. They never update using test labels.</p>'''
    student=ev['held_out_student'];future=ev['forward_time']
    text+=f'''<h2>Interpretation</h2><p><strong>These experiments do not establish reliable predictive value from the environmental measurements.</strong> For unseen participant IDs, every fitted ridge/forest model has higher MAE than the median baseline ({student['Median baseline']['mae']:.3f}). The environment-only forest scores {student['Environment | forest']['mae']:.3f}; the combined forest scores {student['Environment + personal/time | forest']['mae']:.3f}. The primary combined ridge also fails to beat the baseline.</p><p>On later Spring weeks, the combined forest scores {future['Environment + personal/time | forest']['mae']:.3f}, better than the median baseline's {future['Median baseline']['mae']:.3f}, but personal/time without environment scores {future['Personal/time | forest']['mae']:.3f}. Earlier personal-average stress scores are more informative still: MAE {future['Past-person mean baseline']['mae']:.3f}. All14 future-test IDs have matched training history, so this history baseline is available there; it cannot be used for genuinely unseen students. The combined forest's interval for improvement over the mean baseline includes zero. The strongest result is persistence in individual stress scores, not an established environmental effect.</p>'''
    for protocol,models in ev.items():
        text+='<h2>'+protocol+'</h2><table><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>'
        for name,v in models.items():text+='<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[k]:.3f}</td>' for k in ['mae','rmse','r2'])+'</tr>'
        text+='</table>'
    text+='<h2>Uncertainty</h2><p>MAE change below is model minus baseline; negative favors the model. These intervals resample fixed prediction clusters by participant without refitting. They do not include full training, selection or shared time/cohort uncertainty.</p><table><tr><th>Protocol / comparison</th><th>MAE change</th><th>Person-cluster95% interval</th></tr>'
    for protocol,comp in out['paired_intervals'].items():
        for name,v in comp.items():
            low,high=v['person_cluster_95_interval'];text+=f'<tr><td>{protocol}: {html.escape(name)}</td><td>{v["mae_change"]:.3f}</td><td>{low:.3f} to {high:.3f}</td></tr>'
    text+='</table><h2>Limits and outputs</h2><p>'+html.escape(m['mapping_note'])+' The bedroom sensors do not measure participant presence. Environmental extremes, sensor calibration and recording gaps can affect summaries. Only a small subset of the survey cohort has sufficient sensor coverage. Demographic timing is not documented. This pilot follows prior experiments on another dataset; it is not external validation. A baseline or comparator can score well through stable participant or cohort differences rather than sensitivity to changes in stress.</p><p><a href="metrics.json">Full metrics and methods</a>; <a href="predictions.csv">held-out predictions</a>; <a href="modeling_data.csv">prepared weekly data</a>; <a href="exposure_window_audit.csv">exposure coverage audit</a>; <a href="integrity_checks.json">integrity checks</a>. Fitted models and selected feature lists are saved in models.joblib; use the payload bounds0–4 after predicting. These are research artifacts, not a deployed service. Original files and assessment results remain unchanged. Run depress_weekly_stress.py to reproduce.</p></body></html>'
    (OUT/'report.html').write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True);df,scopes,metadata=prepare();df.to_csv(OUT/'modeling_data.csv',index=False);models=specifications(scopes);evaluation={};parts=[]
    print(json.dumps({k:metadata[k] for k in ['n','people','dates','cohort_counts','missing_covariates','primary_model']},indent=2),flush=True)
    for protocol in ['held_out_student','forward_time']:
        scores,pred=evaluate(df,models,protocol);evaluation[protocol]=scores;parts.append(pred);print('Completed',protocol,flush=True)
    predictions=pd.concat(parts,ignore_index=True);predictions.to_csv(OUT/'predictions.csv',index=False)
    tr,te=next(make_splits(df,'forward_time'));train=df.iloc[tr];test=df.iloc[te]
    split={'train_n':len(train),'test_n':len(test),'gap_responses':len(df)-len(train)-len(test),'test_people':int(test.person.nunique()),'seen_test_people':len(set(train.person)&set(test.person)),'train_last':str(train.date.max().date()),'test_first':str(test.date.min().date()),'test_last':str(test.date.max().date()),'test_cohorts':test.study_cohort.value_counts().to_dict()}
    intervals=paired_intervals(predictions)
    fitted={name:clone(template).fit(df[cols],df.PSS) for name,(cols,template) in models.items()}
    payload={'models':fitted,'features':{name:cols for name,(cols,_) in models.items()},'bounds':BOUNDS,'metadata':metadata,'target':'PSS'};joblib.dump(payload,OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    cols=payload['features'][PRIMARY];np.testing.assert_allclose(restored['models'][PRIMARY].predict(df[cols]),fitted[PRIMARY].predict(df[cols]),rtol=0,atol=1e-12)
    assert predictions.predicted.between(*BOUNDS).all() and not predictions.duplicated(['protocol','model','source_row']).any()
    checks={'target_values_match_original_weekly_file':True,'all_234_assessment_windows_reconstructed':True,'no_survey_day_or_future_environment_in_features':True,'seven_day_forward_gap_and_disjoint_exposure_support_verified':True,'student_ids_disjoint_in_each_student_fold':True,'no_target_or_identity_predictors':True,'no_ambiguous_sensor_mapping_forced':True,'source_DST_repeated_hour_retained_with_unique_UTC_instants':True,'final_serialization_predictions_match':True,'all_scored_predictions_bounded_0_to_4':True,'no_duplicate_prediction_keys':True}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2))
    output={'metadata':metadata,'forward_split':split,'evaluation':evaluation,'paired_intervals':intervals};(OUT/'metrics.json').write_text(json.dumps(output,indent=2));report(output)
    print(json.dumps({'forward_split':split,'evaluation':evaluation,'paired_intervals':intervals},indent=2),flush=True)

if __name__=='__main__':main()
