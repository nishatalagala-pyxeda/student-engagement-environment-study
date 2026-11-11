"""All available tabular predictor screen, with explicit outcome-proximity ablations."""
from pathlib import Path
import os,json,hashlib,html
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import broad_environment_model as cls
import valence_regression as reg
import subject_step2 as step2
import demographics_step1 as step1

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/all-tabular-predictors'
LIKERT={'Strongly disagree':-2,'Somewhat disagree':-1,'Neither agree nor disagree':0,'Somewhat agree':1,'Strongly agree':2}
CLOTHING=['Shirt','Jumper','Jacket','Pants','Shorts','Skirt','Dress','Other']
SUBJECT=step2.SUBJECT

def prepare():
    source=ROOT/'outputs/subject-step2/modeling_data.csv'
    df=pd.read_csv(source,parse_dates=['timestamp','feature_end'])
    oldmeta=json.loads((ROOT/'outputs/broad-environment-model/metrics.json').read_text())['metadata']
    data_dir=Path(oldmeta['input_dir']);survey=pd.read_csv(data_dir/'survey/student_survey.csv');students=pd.read_csv('/Users/nishatalagala/Downloads/student.csv')
    source_survey=survey.iloc[(df.source_row-2).astype(int)].reset_index(drop=True)
    assert np.array_equal(source_survey.Pid,df.Pid) and np.array_equal(source_survey.Valence,df.Valence)
    bg=students.set_index('Pid').loc[df.Pid].reset_index()
    assert np.array_equal(bg.Age,df.Age) and np.array_equal(bg.Gender,df.Gender)
    catalog={}
    def note(name,source,availability,meaning):catalog[name]={'source':source,'availability':availability,'meaning':meaning}
    for c in cls.WEATHER_FEATURES+cls.INDOOR_FEATURES:note(c,'Verified environmental preparation','Context','Six-reading trailing average; campus indoor values are fixed-eight-room medians, not student-room exposure.')
    for c in ['Age','recorded_female']:note(c,'Student background','Before survey','Age in years or recorded Female=1/Male=0.')
    note(SUBJECT,'Timetable and class groups','Context','Current scheduled subject or explicit outside/missing schedule category; not attendance confirmation.')
    thermal_cols=list(students.columns[6:9])
    background=['background_thermal_sensation','background_comfort_when_engaged','background_temperature_distraction']
    df[background[0]]=pd.to_numeric(bg[thermal_cols[0]])
    for new,old in zip(background[1:],thermal_cols[1:]):
        df[new]=bg[old].map(LIKERT);assert df[new].notna().all()
    for c in background:note(c,'Student background','Before survey','General thermal sensation (−3…3), comfort or distraction agreement (−2…2).')
    groupcats=['form_group','math_group','language_group']
    for new,old in zip(groupcats,['Form Room','Math Room','Language Room']):
        df[new]=bg[old];note(new,'Student background','Before survey','Class group/room label; not mapped to environmental sensor filename.')
    df['study_week']=source_survey.Week.astype(str)
    df['weekday_sin']=np.sin(2*np.pi*(source_survey.Weekday-1)/7);df['weekday_cos']=np.cos(2*np.pi*(source_survey.Weekday-1)/7)
    times=cls.TIME_FEATURES+['weekday_sin','weekday_cos']
    for c in times+['study_week']:note(c,'Verified survey date/time','Context','Cyclic clock/weekday encoding or categorical study-week index. No exact calendar date ID.')
    audit=pd.read_csv(ROOT/'outputs/demographics-step1/subject_availability.csv').set_index('source_row').loc[df.source_row].reset_index()
    table=pd.read_csv(data_dir/'participant_class_info/class_table.csv').set_index('Class_id')
    df['scheduled_room']='Not in a uniquely scheduled class';df['recent_subject']='No recent-class proxy'
    for c in ['scheduled_duration_minutes','scheduled_class_number','scheduled_class_progress','minutes_since_recent_class']:df[c]=np.nan
    for i,a in audit.iterrows():
        if a.status=='current_scheduled_class':
            row=table.loc[int(a.class_id)];start=pd.to_timedelta(row.Start_time+':00').total_seconds()/60;end=pd.to_timedelta(row.Finish_time+':00').total_seconds()/60
            minute=df.iloc[i].timestamp.hour*60+df.iloc[i].timestamp.minute+df.iloc[i].timestamp.second/60
            df.loc[i,'scheduled_room']=row.Room;df.loc[i,'scheduled_duration_minutes']=end-start;df.loc[i,'scheduled_class_number']=row.Class_no;df.loc[i,'scheduled_class_progress']=(minute-start)/(end-start)
        elif a.status=='recent_class_proxy_only':
            df.loc[i,'recent_subject']=a.scheduled_subject_or_recent_proxy;df.loc[i,'minutes_since_recent_class']=a.minutes_since_previous_class
    contextnums=['scheduled_duration_minutes','scheduled_class_number','scheduled_class_progress','minutes_since_recent_class']
    assert df.scheduled_class_progress.dropna().between(0,1,inclusive='left').all()
    for c in contextnums+['scheduled_room','recent_subject']:note(c,'Timetable availability audit','Context','Uniquely matched current class, or explicitly labeled ≤30-minute recent-class proxy. Missing numeric context stays missing and is imputed in training folds.')
    frames={}
    for room in cls.ROOMS+['19']:
        d=pd.read_csv(data_dir/'longitudinal'/f'{room}.csv');d.index=pd.Timestamp('2019-10-07')+pd.to_timedelta(d.Day-1,unit='D')+pd.to_timedelta(d.Time);frames[room]=d
    extra=frames['19'][['OutdoorDewpoint','OutdoorGustSpeed','UvLevel']].copy()
    direction=np.deg2rad(frames['19'].OutdoorWindDirection)
    extra['OutdoorWindDirectionSin']=np.sin(direction);extra['OutdoorWindDirectionCos']=np.cos(direction)
    for field in ['IndoorCO2','IndoorNoise','CoolingState','HeatingState','Occupied']:
        matrix=pd.concat([frames[r][field].where(frames[r].UsabilityMask) for r in cls.ROOMS],axis=1)
        name='CampusMedian'+field.removeprefix('Indoor') if field.startswith('Indoor') else 'CampusFraction'+field
        extra[name]=(matrix.median(axis=1) if field.startswith('Indoor') else matrix.mean(axis=1)).where(matrix.notna().all(axis=1))
    extra=extra.rolling(6,min_periods=6).mean();extra.columns=[c+'_trailing6_mean' for c in extra]
    for c in extra:
        df[c]=extra.reindex(pd.DatetimeIndex(df.feature_end))[c].to_numpy();assert df[c].notna().all()
        note(c,'Longitudinal outdoor or fixed-eight-room campus panel','Context','Trailing six readings at/before survey. Campus CO₂/noise are medians across rooms; state/occupancy fractions are campus schedule/derived-state proxies, not the student room. Noise is arithmetic mean of reported dB, not energy-equivalent sound level.')
    env=cls.WEATHER_FEATURES+cls.INDOOR_FEATURES+list(extra.columns)
    df['current_thermal_sensation']=source_survey.Thermal_sensation
    df['current_thermal_preference']=source_survey.Thermal_preference
    df['seat_x_fraction']=source_survey.Loc_x/400;df['seat_y_fraction']=source_survey.Loc_y/321
    tokens=source_survey.Clothing.map(lambda v:{x.strip() for x in v.split(',')})
    assert set.union(*tokens.tolist()).issubset(set(CLOTHING))
    cloth=[]
    for token in CLOTHING:
        c='clothing_'+token.lower();df[c]=tokens.map(lambda v:int(token in v));cloth.append(c)
    engagement=[]
    for i in range(1,6):
        c=f'engagement_item_{i}';df[c]=source_survey[f'Engage_{i}'].map(LIKERT);assert df[c].notna().all()
        if i in [2,4]:df[c]=-df[c]
        engagement.append(c)
    df['survey_confidence']=source_survey.Confidence_level
    survey_num=['current_thermal_sensation','seat_x_fraction','seat_y_fraction']+cloth+engagement+['survey_confidence']
    for c in survey_num+['current_thermal_preference']:note(c,'Same mood survey','Same survey; not an independent forecast','Thermal feelings/preference, clothing indicators, seating pixels normalized to thumbnail size, individual engagement responses (items 2/4 reversed), or response confidence. Seat coordinates are not comparable physical distances across rooms.')
    df['pam_arousal']=source_survey.Arousal;assert df.pam_arousal.notna().all()
    note('pam_arousal','Same PAM picture choice as valence','Same-outcome-instrument diagnostic only','Arousal and valence are derived from the same chosen image. This is a label-rich contemporaneous diagnostic, not an independent predictor of future mood.')
    previous_num=cls.WEATHER_FEATURES+cls.INDOOR_FEATURES+step1.DEMOGRAPHICS
    background_num=previous_num+background;background_cat=[SUBJECT]+groupcats
    context_num=env+step1.DEMOGRAPHICS+background+times+contextnums
    context_cat=[SUBJECT]+groupcats+['study_week','scheduled_room','recent_subject']
    full_num=context_num+survey_num;full_cat=context_cat+['current_thermal_preference']
    scopes={
        'Previous subject model':(previous_num,[SUBJECT]),
        'Background added':(background_num,background_cat),
        'All context':(context_num,context_cat),
        'Context without environment':([c for c in context_num if c not in env],context_cat),
        'All non-mood tabular':(full_num,full_cat),
        'Non-mood without environment':([c for c in full_num if c not in env],full_cat),
        'Same-PAM diagnostic':(full_num+['pam_arousal'],full_cat),
    }
    for scope,(nums,cats) in scopes.items():
        assert not set(nums+cats).intersection({'Pid','participant','source_row','Class_id','Valence','higher_valence','date','timestamp'})
        assert 'pam_arousal' not in nums or scope=='Same-PAM diagnostic'
        assert len(nums+cats)==len(set(nums+cats))
    return df,scopes,catalog,env,source,data_dir

def build_model(task,nums,cats,forest=False):
    estimator=clone((cls.specifications() if task=='classification' else reg.specifications())['Environment forest' if forest else ('Environment logistic' if task=='classification' else 'Environment ridge')][1].steps[-1][1])
    numeric=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    if not forest:numeric=make_pipeline(numeric,StandardScaler())
    preprocessor=ColumnTransformer([('numeric',numeric,nums),('categorical',make_pipeline(SimpleImputer(strategy='constant',fill_value='Unknown'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),cats)],remainder='drop')
    return make_pipeline(preprocessor,estimator)

def specs_for(task,scopes):
    bases=cls.specifications() if task=='classification' else reg.specifications()
    names=['Majority baseline','Time of day logistic'] if task=='classification' else ['Mean baseline','Median baseline','Time of day ridge']
    result={n:bases[n] for n in names}
    for scope,(nums,cats) in scopes.items():
        result[scope+' | linear']=(nums+cats,build_model(task,nums,cats))
        if scope in ['Previous subject model','All context','All non-mood tabular','Same-PAM diagnostic']:
            result[scope+' | forest']=(nums+cats,build_model(task,nums,cats,True))
    return result

def check_old_predictions(pred,task):
    old=pd.read_csv(ROOT/'outputs/subject-step2'/(task+'_predictions.csv'))
    oldname=('Environment logistic' if task=='classification' else 'Environment ridge')+' + age/gender + subject'
    for new,oldname in [('Previous subject model | linear',oldname),('Previous subject model | forest','Environment forest + age/gender + subject')]:
        a=pred[pred.model.eq(new)].set_index(['protocol','source_row']).sort_index();b=old[old.model.eq(oldname)].set_index(['protocol','source_row']).sort_index()
        assert a.index.equals(b.index)
        col='probability_higher' if task=='classification' else 'predicted_valence';np.testing.assert_allclose(a[col],b[col],rtol=0,atol=1e-9)

def report(output):
    evaluation=output['evaluation'];scopes=output['metadata']['scope_features']
    def table(task,protocol):
        keys=['accuracy','balanced_accuracy','roc_auc'] if task=='classification' else ['mae','rmse','r2']
        names=['Accuracy ↑','Balanced accuracy ↑','ROC AUC ↑'] if task=='classification' else ['MAE ↓','RMSE ↓','R² ↑']
        rows=''.join('<tr><td>'+html.escape(n)+'</td>'+''.join(f'<td>{m[k]:.3f}</td>' for k in keys)+'</tr>' for n,m in evaluation[task][protocol].items())
        return '<table><tr><th>Model/scope</th>'+''.join('<th>'+n+'</th>' for n in names)+'</tr>'+rows+'</table>'
    text='''<!doctype html><html><head><meta charset="utf-8"><title>All tabular predictors</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}code{overflow-wrap:anywhere}details{margin:15px 0}</style></head><body><h1>All available tabular predictors: valence classification and regression</h1><p>Exploratory feature expansion, November 10, 2026. Same 163 observations, 22 students, nine dates and six excluded conflicting responses as before. Original data and earlier models remain unchanged.</p>
    <h2>Three different research questions</h2><ol><li><strong>Background/context prediction:</strong> broad environment, demographics, thermal-background answers, class groups, schedule and time. These do not require the current mood-survey answers.</li><li><strong>Contemporaneous non-mood questionnaire:</strong> additionally use current thermal feelings/preferences, clothing, seating, engagement and confidence. These can be useful correlates but are answered alongside mood; gains do not establish environmental prediction or future-mood forecasting.</li><li><strong>Same-PAM diagnostic:</strong> also use arousal from the same image choice that generates valence. This is intentionally a separate outcome-proximity diagnostic and must not be claimed as independent mood prediction.</li></ol>
    <p>Valence, its binary encoding, participant ID, class ID, source row and exact-date identifiers never enter the predictors. Teacher attributes cannot be reliably assigned to student responses from the supplied metadata. Raw wearable physiology requires a separate signal-preparation analysis and is not included in this tabular pass.</p>
    <h2>Inputs added</h2><p>Three thermal-background responses; form/maths/language group labels; study-week and cyclic weekday/clock time; current scheduled room, class duration/number/progress, recent-class subject and gap when explicitly available; remaining outdoor dew point/gust/UV/circular wind-direction variables; campus CO₂/noise medians and cooling/heating/occupied fractions; current thermal sensation/preference; eight clothing indicators; two normalized seating coordinates; five individually coded engagement answers; survey confidence; and arousal only in its diagnostic scope.</p>
    <p>All campus measurements use the same fixed eight-room panel and six-reading trailing averaging ending at/before the survey. CO₂ and noise are campus context proxies, not the student's room; saturation and calibration concerns remain. Cooling/heating are inferred states, occupied is schedule-derived, and noise summaries average reported dB rather than acoustic energy. Known-invalid longitudinal LessonPct is excluded; class progress is recomputed from matched En-Gage class times. No environmental filename mapping is invented.</p>
    <h2>Feature sets</h2>'''
    for scope,features in scopes.items():text+='<details><summary>'+html.escape(scope)+' — '+str(len(features))+' input fields before encoding</summary><p>'+html.escape(', '.join(features))+'</p></details>'
    text+='''<p>Room/group/subject/preferences are categorical, with one-hot encoders fitted only on training folds. Likert agreements map to −2…2; engagement items 2 and 4 are reversed. Clothing is encoded as separate fixed answer-option flags. Missing numeric context/confidence uses training-median imputation plus missing indicators. All-empty training columns are retained safely. No global imputation, tuning or feature selection uses test outcomes.</p>
    <h2>Validation</h2><p>Hold out complete dates; train on October and test on later November; and hold out complete students as a supplemental test. Students may occur on other training dates in date tests; dates are shared in student tests. Hyperparameters remain logistic C=1, ridge alpha=1, forests 300 trees/depth 3/minimum leaf 10, seed 42. No search for a best parameter setting. Old subject-model predictions are reproduced. Linear models cover all scopes; forests cover the previous, context, non-mood and same-PAM scopes.</p><p>Scores below are sequential exploratory results after earlier results were inspected. Nine dates are not enough to declare a robust best model. Any selected winner needs fresh prospective evaluation; no causal demographic, subject or environmental effects are established.</p>'''
    for protocol in ['held_out_date','future_block','held_out_student']:
        for task in ['classification','regression']:text+=f'<h2>{protocol}: {task}</h2>'+table(task,protocol)
    text+='''<h2>Uncertainty and attribution</h2><p>Paired changes in metrics.json resample existing held-out prediction date/student clusters without refitting; small cluster counts and crossed student/date dependencies limit inference. Compare context with versus without environment, and the non-mood sets with versus without environment, to distinguish environmental information from student/survey information. The arousal diagnostic is not the primary independently available-context model.</p><h2>Files</h2><p><a href="metrics.json">Full metrics and paired comparisons</a>; <a href="feature_catalog.json">every feature, source and availability</a>; <a href="modeling_data.csv">prepared observations</a>; <a href="classification_predictions.csv">held-out classification</a>; <a href="regression_predictions.csv">held-out regression</a>. Separate fitted models store required columns and trained encoders. Regression outputs require clipping to the stored bounds [1,4]. Run all_predictors.py to reproduce. This pass is complete for available table predictors; wearable preprocessing and new clinical/causal claims are outside these results.</p></body></html>'''
    (DEST/'report.html').write_text(text)

def main():
    DEST.mkdir(parents=True,exist_ok=True);df,scopes,catalog,env,source,data_dir=prepare();df.to_csv(DEST/'modeling_data.csv',index=False)
    (DEST/'feature_catalog.json').write_text(json.dumps(catalog,indent=2));results={};predictions={};payloads={};pairs={}
    comparisons=[('Background added | linear','Previous subject model | linear'),('All context | linear','Background added | linear'),('All context | linear','Context without environment | linear'),('All non-mood tabular | linear','All context | linear'),('All non-mood tabular | linear','Non-mood without environment | linear'),('Same-PAM diagnostic | linear','All non-mood tabular | linear')]
    for task in ['classification','regression']:
        specs=specs_for(task,scopes);results[task]={};parts=[]
        for protocol in ['held_out_date','future_block','held_out_student']:
            scores,p=(cls.run_protocol if task=='classification' else reg.evaluate)(df,specs,protocol);results[task][protocol]=scores;parts.append(p);print('Completed',task,protocol,flush=True)
        pred=pd.concat(parts,ignore_index=True);check_old_predictions(pred,task);pred.to_csv(DEST/(task+'_predictions.csv'),index=False);predictions[task]=pred;pairs[task]={}
        for protocol in ['held_out_date','future_block','held_out_student']:
            pairs[task][protocol]={}
            for new,old in comparisons:
                v=step1.paired_interval(pred,task,protocol,new,old);v['unit']=v['unit'].replace('demographics','the first-named feature set');pairs[task][protocol][new+' vs '+old]=v
        target='higher_valence' if task=='classification' else 'Valence'
        fitted={n:clone(m).fit(df[cols],df[target]) for n,(cols,m) in specs.items()}
        payloads[task]={'models':fitted,'features':{n:cols for n,(cols,_) in specs.items()},'target':target,'bounds':(1,4) if task=='regression' else None,'warning':'Same-PAM diagnostic uses the valence-label image choice; same-survey predictors are contemporaneous, not independent forecasts.'}
    metadata={'n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'scope_features':{k:nums+cats for k,(nums,cats) in scopes.items()},'environment_features':env,'missing_counts':df[list(catalog)].isna().sum().to_dict(),'current_survey_features':[k for k,v in catalog.items() if v['availability'].startswith('Same survey')],'same_pam_feature':'pam_arousal','wearables_included':False,'baseline_predictions_reproduced':True,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'student_sha256':hashlib.sha256(Path('/Users/nishatalagala/Downloads/student.csv').read_bytes()).hexdigest(),'survey_sha256':hashlib.sha256((data_dir/'survey/student_survey.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for task,payload in payloads.items():
        payload['metadata']=metadata;joblib.dump(payload,DEST/(task+'_models.joblib'));restored=joblib.load(DEST/(task+'_models.joblib'));name='All context | linear';cols=payload['features'][name]
        np.testing.assert_allclose(restored['models'][name].predict(df[cols]),payload['models'][name].predict(df[cols]))
    output={'metadata':metadata,'evaluation':results,'paired_changes':pairs};(DEST/'metrics.json').write_text(json.dumps(output,indent=2));report(output);print(json.dumps({'metadata':metadata,'evaluation':results},indent=2),flush=True)

if __name__=='__main__':main()
