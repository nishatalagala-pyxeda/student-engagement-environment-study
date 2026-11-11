"""Thermal sensation/preference: fixed exploratory models, no current comfort leakage."""
from pathlib import Path
import os,json,zipfile,hashlib,html
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,mean_absolute_error,mean_squared_error,r2_score
import broad_environment_model as envmod
import demographics_step1 as demo
import all_predictors as tab
import wearable_predictors as wrist
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/thermal-models'
PROTOCOLS=['held_out_date','future_block','held_out_student']


def prepare():
    data_dir=Path(json.loads((ROOT/'outputs/broad-environment-model/metrics.json').read_text())['metadata']['input_dir'])
    with zipfile.ZipFile('/Users/nishatalagala/Downloads/Figshare-Navya/Survey.zip') as z:
        raw=pd.read_csv(z.open('Survey/Student_survey.csv'))
    local=pd.read_csv(data_dir/'survey/student_survey.csv')
    common=[c for c in raw if c!='Time']
    assert raw[common].fillna('__NA__').equals(local[common].fillna('__NA__'))
    df=local.copy();df['source_row']=df.index+2;df['timestamp']=pd.to_datetime(raw.Time,dayfirst=True)
    assert (pd.to_timedelta(df.timestamp.dt.strftime('%H:%M:%S'))==pd.to_timedelta(local.Time)).all()
    df['date']=df.timestamp.dt.strftime('%Y-%m-%d');df['feature_end']=df.timestamp.dt.floor('5min')
    assert df.Thermal_sensation.isin(range(-3,4)).all()
    assert df.Thermal_preference.isin(['Cooler','No change','Warmer']).all()
    # Clean on the two new outcomes, independently of whether mood is available.
    groups=df.groupby(['Pid','timestamp'])
    conflict=(groups.Thermal_sensation.transform('nunique')>1)|(groups.Thermal_preference.transform('nunique')>1)
    rejected=df[conflict].copy();clean=df[~conflict].drop_duplicates(['Pid','timestamp']).copy()
    audit={'raw_responses':len(raw),'conflicting_comfort_rows':int(conflict.sum()),'redundant_same_time_rows_removed':int((~conflict).sum()-len(clean))}
    frames={}
    for room in envmod.ROOMS+['19']:
        d=pd.read_csv(data_dir/'longitudinal'/f'{room}.csv');d.index=pd.Timestamp('2019-10-07')+pd.to_timedelta(d.Day-1,unit='D')+pd.to_timedelta(d.Time);frames[room]=d
    env=frames['19'][envmod.OUTDOOR+['OutdoorDewpoint','OutdoorGustSpeed','UvLevel']].copy()
    direction=np.deg2rad(frames['19'].OutdoorWindDirection)
    env['OutdoorWindDirectionSin']=np.sin(direction);env['OutdoorWindDirectionCos']=np.cos(direction)
    for field in ['IndoorTemperature','IndoorHumidity','IndoorCO2','IndoorNoise','CoolingState','HeatingState','Occupied']:
        matrix=pd.concat([frames[r][field].where(frames[r].UsabilityMask) for r in envmod.ROOMS],axis=1)
        name='CampusMedian'+field.removeprefix('Indoor') if field.startswith('Indoor') else 'CampusFraction'+field
        env[name]=(matrix.median(axis=1) if field.startswith('Indoor') else matrix.mean(axis=1)).where(matrix.notna().all(axis=1))
    env=env.rolling(6,min_periods=6).mean();env.columns=[c+'_trailing6_mean' for c in env]
    for c in env:clean[c]=env.reindex(pd.DatetimeIndex(clean.feature_end))[c].to_numpy()
    covered=clean[list(env)].notna().all(axis=1)
    excluded=clean[~covered].copy();df=clean[covered].reset_index(drop=True)
    audit.update({'clean_comfort_responses':len(clean),'excluded_no_complete_environment':int((~covered).sum()),'n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'date_counts':df.date.value_counts().sort_index().to_dict()})
    students=pd.read_csv('/Users/nishatalagala/Downloads/student.csv');bg=students.set_index('Pid').loc[df.Pid].reset_index()
    df['Age']=bg.Age;df['recorded_female']=bg.Gender.map({'Female':1,'Male':0})
    backgrounds=['background_thermal_sensation','background_comfort_when_engaged','background_temperature_distraction']
    for i,c in enumerate(backgrounds):df[c]=pd.to_numeric(bg.iloc[:,6+i]) if i==0 else bg.iloc[:,6+i].map(tab.LIKERT)
    assert df[backgrounds+['Age','recorded_female']].notna().all().all()
    groupcats=['form_group','math_group','language_group']
    for c,old in zip(groupcats,['Form Room','Math Room','Language Room']):df[c]=bg[old]
    minute=df.timestamp.dt.hour*60+df.timestamp.dt.minute
    df['clock_sin']=np.sin(2*np.pi*minute/1440);df['clock_cos']=np.cos(2*np.pi*minute/1440)
    df['weekday_sin']=np.sin(2*np.pi*(df.Weekday-1)/7);df['weekday_cos']=np.cos(2*np.pi*(df.Weekday-1)/7);df['study_week']=df.Week.astype(str)
    schedule,_=demo.subject_audit(df,students,data_dir)
    df['scheduled_subject_context']='Outside scheduled class';df['scheduled_room']='Not in a uniquely scheduled class'
    df['recent_subject']='No recent-class proxy'
    timing=['scheduled_duration_minutes','scheduled_class_number','scheduled_class_progress','minutes_since_recent_class']
    for c in timing:df[c]=np.nan
    table=pd.read_csv(data_dir/'participant_class_info/class_table.csv').set_index('Class_id')
    for i,a in schedule.iterrows():
        if a.status=='current_scheduled_class':
            row=table.loc[int(a.class_id)];start=pd.to_timedelta(row.Start_time+':00').total_seconds()/60;end=pd.to_timedelta(row.Finish_time+':00').total_seconds()/60
            now=df.iloc[i].timestamp.hour*60+df.iloc[i].timestamp.minute+df.iloc[i].timestamp.second/60
            df.loc[i,'scheduled_subject_context']=row.Subject;df.loc[i,'scheduled_room']=row.Room
            df.loc[i,timing[:3]]=[end-start,row.Class_no,(now-start)/(end-start)]
        elif a.status=='recent_class_proxy_only':df.loc[i,'recent_subject']=a.scheduled_subject_or_recent_proxy;df.loc[i,'minutes_since_recent_class']=a.minutes_since_previous_class
        elif a.status=='no_eligible_timetable_for_day':df.loc[i,'scheduled_subject_context']='No timetable entry'
        elif a.status=='before_first_scheduled_class':df.loc[i,'scheduled_subject_context']='Before classes'
        elif 'ambiguous' in a.status:df.loc[i,'scheduled_subject_context']='Ambiguous schedule'
    clothing=[]
    tokens=df.Clothing.map(lambda v:{x.strip() for x in v.split(',')})
    assert set.union(*tokens.tolist()).issubset(set(tab.CLOTHING))
    for token in tab.CLOTHING:
        c='clothing_'+token.lower();df[c]=tokens.map(lambda v:int(token in v));clothing.append(c)
    df['seat_x_fraction']=df.Loc_x/400;df['seat_y_fraction']=df.Loc_y/321
    nums=list(env)+['Age','recorded_female']+backgrounds+envmod.TIME_FEATURES+['weekday_sin','weekday_cos']+timing
    cats=['scheduled_subject_context']+groupcats+['study_week','scheduled_room','recent_subject']
    scopes={'Environment':(list(env),[]),'Context without environment':([c for c in nums if c not in env],cats),'Context':(nums,cats),'Context + clothing/seating':(nums+clothing+['seat_x_fraction','seat_y_fraction'],cats)}
    # Reuse verified feature extraction for existing observations; extract only new comfort responses.
    cached=pd.read_csv(ROOT/'outputs/all-predictors-with-wearables/modeling_data.csv')
    wear=[c for c in cached if c.startswith('wear_')]
    cached=cached[['Pid','timestamp']+wear];cached['timestamp']=pd.to_datetime(cached.timestamp)
    df=df.merge(cached,on=['Pid','timestamp'],how='left',validate='one_to_one',indicator=True)
    missing=df._merge.eq('left_only');audit['new_wearable_windows_to_extract']=int(missing.sum())
    if missing.any():
        features,assignment,extraction=wrist.extract(df[missing])
        features=features.set_index('source_row')
        df.loc[missing,wear]=features.reindex(df.loc[missing,'source_row'])[wear].to_numpy()
        assignment.to_csv(DEST/'new_recording_assignments.csv',index=False)
        (DEST/'new_wearable_extraction.json').write_text(json.dumps(extraction,indent=2))
    df=df.drop(columns='_merge');scopes['Context + wearables']=(nums+wear,cats)
    scopes['Context + clothing/seating + wearables']=(nums+clothing+['seat_x_fraction','seat_y_fraction']+wear,cats)
    df['sensation_class']=np.select([df.Thermal_sensation.lt(0),df.Thermal_sensation.gt(0)],['Cold','Warm'],default='Neutral')
    audit['sensation_counts']=df.Thermal_sensation.value_counts().sort_index().to_dict();audit['sensation_class_counts']=df.sensation_class.value_counts().to_dict();audit['preference_counts']=df.Thermal_preference.value_counts().to_dict()
    assert not df.duplicated(['Pid','timestamp']).any()
    for nums,cats in scopes.values():assert not set(nums+cats).intersection({'Thermal_sensation','Thermal_preference','Valence','Arousal','Pid','source_row','date','sensation_class'})
    rejected.to_csv(DEST/'excluded_conflicting_comfort.csv',index=False);excluded.to_csv(DEST/'excluded_environment_unavailable.csv',index=False);schedule.to_csv(DEST/'schedule_audit.csv',index=False)
    return df,scopes,audit


def splits(df,protocol):
    if protocol=='held_out_date':pairs=[(df.date.ne(v),df.date.eq(v)) for v in sorted(df.date.unique())]
    elif protocol=='held_out_student':pairs=[(df.Pid.ne(v),df.Pid.eq(v)) for v in sorted(df.Pid.unique())]
    else:pairs=[(df.timestamp.lt('2019-11-02'),df.timestamp.ge('2019-11-18'))]
    for train,test in pairs:
        assert train.any() and test.any() and not (train&test).any()
        if protocol=='held_out_date':assert set(df.loc[train,'date']).isdisjoint(df.loc[test,'date'])
        elif protocol=='held_out_student':assert set(df.loc[train,'Pid']).isdisjoint(df.loc[test,'Pid'])
        else:assert df.loc[train,'timestamp'].max()<df.loc[test,'timestamp'].min()
        yield np.flatnonzero(train),np.flatnonzero(test)


def specs(scopes,regression):
    result={}
    if regression:
        result['Mean baseline']=(envmod.TIME_FEATURES,DummyRegressor(strategy='mean'));result['Median baseline']=(envmod.TIME_FEATURES,DummyRegressor(strategy='median'))
    else:result['Majority baseline']=(envmod.TIME_FEATURES,DummyClassifier(strategy='most_frequent'))
    for name,(nums,cats) in scopes.items():
        for forest,family in [(False,'linear'),(True,'forest')]:
            result[name+' | '+family]=(nums+cats,tab.build_model('regression' if regression else 'classification',nums,cats,forest))
    return result


def evaluate(df,models,target,regression,protocol,bounds=(-3,3)):
    y=df[target].to_numpy();labels=sorted(df[target].unique());scores={};records=[]
    for name,(cols,template) in models.items():
        predicted=np.full(len(df),np.nan) if regression else np.full(len(df),'',dtype=object)
        for train,test in splits(df,protocol):
            model=clone(template).fit(df.iloc[train][cols],y[train]);p=model.predict(df.iloc[test][cols]);predicted[test]=np.clip(p,*bounds) if regression else p
        used=np.isfinite(predicted) if regression else predicted!='';actual=y[used];p=predicted[used]
        if regression:score={'n':len(actual),'mae':float(mean_absolute_error(actual,p)),'rmse':float(np.sqrt(mean_squared_error(actual,p))),'r2':float(r2_score(actual,p)),'bias':float(np.mean(p-actual))}
        else:score={'n':len(actual),'accuracy':float(accuracy_score(actual,p)),'balanced_accuracy':float(balanced_accuracy_score(actual,p)),'macro_f1':float(f1_score(actual,p,labels=labels,average='macro',zero_division=0)),'labels':labels,'confusion_matrix':confusion_matrix(actual,p,labels=labels).tolist()}
        scores[name]=score
        for i,value in zip(np.flatnonzero(used),p):records.append({'protocol':protocol,'model':name,'source_row':int(df.iloc[i].source_row),'participant':int(df.iloc[i].Pid),'date':df.iloc[i].date,'observed':y[i],'predicted':value})
    return scores,pd.DataFrame(records)


def paired(pred,target,regression):
    result={};baseline='Median baseline' if regression else 'Majority baseline'
    for protocol in PROTOCOLS:
        p=pred[pred.protocol.eq(protocol)];base=p[p.model.eq(baseline)].set_index('source_row');result[protocol]={}
        for name in p.model.unique():
            a=p[p.model.eq(name)].set_index('source_row').loc[base.index]
            delta=(np.abs(a.predicted-a.observed)-np.abs(base.predicted-base.observed)) if regression else ((a.predicted==a.observed).astype(float)-(base.predicted==base.observed).astype(float))
            key=a.participant if protocol=='held_out_student' else a.date;clusters=[delta[key.eq(v)].to_numpy() for v in sorted(key.unique())];rng=np.random.default_rng(42)
            boot=[np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))]).mean() for _ in range(3000)]
            result[protocol][name]={'difference':float(delta.mean()),'cluster_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),'clusters':len(clusters),'unit':'MAE change; negative favors model' if regression else 'Accuracy change; positive favors model'}
    return result


def report(output):
    meta=output['metadata'];ev=output['evaluation']
    text='''<!doctype html><html><head><meta charset="utf-8"><title>Thermal sensation and preference</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}</style></head><body><h1>Predicting thermal sensation and preference</h1><p>Exploratory analysis, November 10, 2026.</p>'''
    text+=f'''<p>There are {meta['raw_responses']} comfort responses, including surveys without mood labels. After excluding {meta['conflicting_comfort_rows']} rows with conflicting same-student/time comfort responses and collapsing {meta['redundant_same_time_rows_removed']} redundant same-time responses, {meta['clean_comfort_responses']} remain. The environmental recordings do not cover {meta['excluded_no_complete_environment']} of those responses (September), leaving <strong>{meta['n']} responses, {meta['students']} students and {meta['dates']} dates</strong> for every model in this comparison. Responses are not selected by mood-label availability. Exclusion audits are saved.</p>
    <h2>Targets and inputs</h2><p>Thermal sensation: Cold (−3…−1), Neutral (0), Warm (1…3), plus regression on the original −3…3 ordinal scale. Thermal preference: Cooler, No change, Warmer (three classes). Regression treats sensation scale steps as approximately equal. No comfort target or alternate current comfort answer, mood/arousal, engagement or response-confidence answer is a predictor. Participant, session and exact-date identifiers are excluded.</p><p>Environment uses 17 trailing outdoor/campus fields; indoor measurements are fixed-eight-room medians or state fractions, not student-room exposure. Context adds age, recorded gender, three pre-study thermal-background responses, class groups, scheduled subject/room, class timing and cyclic clock/weekday/study week. Separate models add reported clothing/seating (available only when supplied) and/or 44 pre-survey wearable signal/quality summaries. Current subject is scheduled activity, not verified attendance. See <a href="scope_features.json">exact predictor lists</a>.</p><p>All models share held-out-date, October-to-later-November, and held-out-student splits. All preprocessing is fitted only on training folds. Classifiers use logistic regression C=1 or the fixed 300-tree depth-3/minimum-leaf-10 forest; regressors use ridge alpha=1 or the same fixed forest settings as the prior experiments. No tuning search; sensation predictions are clipped to −3…3. Majority, mean and median baselines are fitted separately inside each fold. The context forest is the primary model, and clothing/wearable additions are comparisons.</p>'''
    for target,protocols in ev.items():
        text+='<h2>'+html.escape(target)+'</h2>'
        for protocol,models in protocols.items():
            keys=['mae','rmse','r2'] if target=='sensation_regression' else ['accuracy','balanced_accuracy','macro_f1']
            text+='<h3>'+protocol+'</h3><table><tr><th>Model</th>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
            for name,v in models.items():text+='<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[k]:.3f}</td>' for k in keys)+'</tr>'
            text+='</table>'
    text+='''<h2>Limits and files</h2><p>Only nine dates support environmental generalization. Students and dates form crossed dependencies: date holdouts retain known students, and student holdouts retain shared dates. These are exploratory follow-on experiments. Missing classroom mapping, self-reported outcomes, timetable uncertainty and wearable clock/artifact/missingness limitations remain. Clothing/seating models require contemporaneous inputs; these are not future forecasts. Wearable extraction uses strictly pre-survey 30-minute windows, Melbourne daylight saving, one selected recording and at least five minutes of eligible uniform-signal data for summaries. Reused cached windows retain the same methods and quality checks. Missingness can predict recording availability rather than physiology. Bootstrap intervals resample fixed held-out date or student prediction clusters without refitting; they do not capture full crossed-dependency or model-selection uncertainty. No causal claim is justified.</p><p><a href="metrics.json">Metrics, counts, confusion matrices and paired intervals</a>; <a href="modeling_data.csv">prepared observations</a>; <a href="scope_features.json">feature lists</a>. Per-target held-out predictions and final fitted pipelines are saved alongside this report. Source data and earlier outputs are unchanged. Run thermal_models.py to reproduce.</p></body></html>'''
    (DEST/'report.html').write_text(text)


def main():
    DEST.mkdir(parents=True,exist_ok=True);df,scopes,metadata=prepare();df.to_csv(DEST/'modeling_data.csv',index=False)
    features={name:nums+cats for name,(nums,cats) in scopes.items()};(DEST/'scope_features.json').write_text(json.dumps(features,indent=2))
    metadata.update({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'primary_model':'Context | forest','wearable_methods_reference':'../all-predictors-with-wearables/wearable_extraction.json','target_leakage_assertions_passed':True,'scope_features':features})
    print(json.dumps({k:v for k,v in metadata.items() if k!='scope_features'},indent=2),flush=True)
    evaluation={};intervals={}
    for name,target,regression in [('sensation_classification','sensation_class',False),('sensation_regression','Thermal_sensation',True),('preference_classification','Thermal_preference',False)]:
        models=specs(scopes,regression);evaluation[name]={};parts=[]
        for protocol in PROTOCOLS:
            scores,pred=evaluate(df,models,target,regression,protocol);evaluation[name][protocol]=scores;parts.append(pred);print('Completed',name,protocol,flush=True)
        pred=pd.concat(parts,ignore_index=True);pred.to_csv(DEST/(name+'_predictions.csv'),index=False);intervals[name]=paired(pred,target,regression)
        fitted={n:clone(model).fit(df[cols],df[target]) for n,(cols,model) in models.items()}
        payload={'models':fitted,'features':{n:cols for n,(cols,_) in models.items()},'target':target,'bounds':(-3,3) if regression else None,'metadata':metadata}
        filename=DEST/(name+'_models.joblib');joblib.dump(payload,filename);restored=joblib.load(filename);cols=payload['features']['Context | forest']
        np.testing.assert_array_equal(restored['models']['Context | forest'].predict(df[cols]),fitted['Context | forest'].predict(df[cols]))
    output={'metadata':metadata,'evaluation':evaluation,'paired_against_baseline':intervals}
    (DEST/'metrics.json').write_text(json.dumps(output,indent=2));report(output);print('All thermal models complete.',flush=True)

if __name__=='__main__':main()
