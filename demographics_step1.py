"""Step 1 only: add age and recorded gender; audit subject availability separately."""
from pathlib import Path
import os,json,hashlib,html
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
import broad_environment_model as classification
import valence_regression as regression

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/demographics-step1'
DEMOGRAPHICS=['Age','recorded_female']

def subject_audit(df,students,data_dir):
    """Keep exact current scheduled class separate from a recent-class proxy."""
    surveys=pd.read_csv(data_dir/'survey/student_survey.csv')
    table=pd.read_csv(data_dir/'participant_class_info/class_table.csv')
    students=students.set_index('Pid');records=[]
    for _,observation in df.iterrows():
        source=surveys.iloc[int(observation.source_row)-2]
        assert source.Pid==observation.Pid and (source.Valence==observation.Valence or (pd.isna(source.Valence) and pd.isna(observation.Valence)))
        student=students.loc[observation.Pid]
        day=table[table.Week.eq(source.Week)&table.Weekday.eq(source.Weekday)].copy()
        eligible=(day.Subject.eq('Language')&day.Room.eq(student['Language Room'])) | (day.Subject.eq('Maths')&day.Room.eq(student['Math Room'])) | day.Subject.isin(['Assembly','Chapel']) | (day.Subject.isin(['Form','English','Science','Politics','PE','Health'])&day.is_Form.eq(student['Form Room']))
        day=day[eligible].copy()
        minute=observation.timestamp.hour*60+observation.timestamp.minute+observation.timestamp.second/60
        start=pd.to_timedelta(day.Start_time+':00').dt.total_seconds()/60
        end=pd.to_timedelta(day.Finish_time+':00').dt.total_seconds()/60
        current=day[(start<=minute)&(minute<end)]
        status='';subject=None;class_id=None;gap=None
        if len(current)==1:
            status='current_scheduled_class';subject=current.iloc[0].Subject;class_id=int(current.iloc[0].Class_id)
        elif len(current)>1:status='ambiguous_schedule'
        elif day.empty:status='no_eligible_timetable_for_day'
        else:
            past=day[end<=minute]
            if len(past):
                ends=pd.to_timedelta(past.Finish_time+':00').dt.total_seconds()/60
                most_recent=ends.max();recent=past[ends.eq(most_recent)];gap=float(minute-most_recent)
                if len(recent)==1 and gap<=30:
                    status='recent_class_proxy_only';subject=recent.iloc[0].Subject;class_id=int(recent.iloc[0].Class_id)
                elif len(recent)>1:status='ambiguous_previous_class'
                else:status='outside_class_or_long_gap'
            else:status='before_first_scheduled_class'
        records.append({'source_row':int(observation.source_row),'participant':int(observation.Pid),'date':observation.date,'timestamp':observation.timestamp.isoformat(),'status':status,'scheduled_subject_or_recent_proxy':subject,'class_id':class_id,'minutes_since_previous_class':gap})
    result=pd.DataFrame(records)
    summary={'status_counts':result.status.value_counts().to_dict(),'current_class_subjects':result[result.status.eq('current_scheduled_class')].scheduled_subject_or_recent_proxy.value_counts().to_dict(),'all_timetable_subjects':sorted(table.Subject.unique()),'warning':'Subjects audited only; no subject feature added. Scheduled subject does not prove attendance or survey location. Recent-class proxy uses at most 30 minutes after a scheduled finish, with no active class.'}
    return result,summary

def extend_specs(spec,baseline_names,linear_name):
    selected={n:spec[n] for n in baseline_names+[linear_name,'Environment forest']}
    for name in [linear_name,'Environment forest']:
        cols,model=spec[name]
        selected[name+' + age/gender']=(cols+DEMOGRAPHICS,clone(model))
    cols,model=spec[linear_name]
    selected['Age/gender only']=(DEMOGRAPHICS,clone(model))
    return selected

def paired_interval(pred,task,protocol,new,old):
    p=pred[pred.protocol.eq(protocol)];a=p[p.model.eq(new)].set_index('source_row');b=p[p.model.eq(old)].set_index('source_row').loc[a.index]
    if task=='classification':
        delta=(a.predicted==a.observed).astype(float)-(b.predicted==b.observed).astype(float)
    else:
        delta=np.abs(a.predicted_valence-a.observed_valence)-np.abs(b.predicted_valence-b.observed_valence)
    key=a.participant if protocol=='held_out_student' else a.date
    values=[delta[key.eq(v)].to_numpy() for v in sorted(key.unique())];rng=np.random.default_rng(42)
    draws=[np.concatenate([values[i] for i in rng.integers(0,len(values),len(values))]).mean() for _ in range(3000)]
    return {'difference':float(delta.mean()),'cluster_bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist(),'clusters':len(values),'unit':'accuracy fraction, positive favors demographics' if task=='classification' else 'MAE points, negative favors demographics','note':'Resamples existing held-out prediction clusters without refitting. Small cluster counts and repeated student/date structure limit inference.'}

def compare_saved_predictions(current,task):
    folder='broad-environment-model' if task=='classification' else 'valence-regression'
    prior=pd.read_csv(ROOT/'outputs'/folder/'evaluation_predictions.csv')
    name='Environment logistic' if task=='classification' else 'Environment ridge'
    value='probability_higher' if task=='classification' else 'predicted_valence'
    names=[name,'Environment forest']
    for model in names:
        a=current[current.model.eq(model)].set_index(['protocol','source_row']).sort_index()
        b=prior[prior.model.eq(model)].set_index(['protocol','source_row']).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[value],b[value],rtol=0,atol=1e-9)

def make_report(metrics):
    def table(task,protocol):
        columns=['accuracy','balanced_accuracy','macro_f1'] if task=='classification' else ['mae','rmse','r2']
        headers=['Accuracy','Balanced accuracy','Macro F1'] if task=='classification' else ['MAE ↓','RMSE ↓','R² ↑']
        rows=[]
        for name,v in metrics['evaluation'][task][protocol].items():
            rows.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[c]:.3f}</td>' for c in columns)+'</tr>')
        return '<table><tr><th>Input/model</th>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join(rows)+'</table>'
    subjects=metrics['subject_audit'];counts=subjects['status_counts'];m=metrics['metadata']
    summary_rows=[]
    for protocol in ['held_out_date','future_block','held_out_student']:
        c=metrics['evaluation']['classification'][protocol];r=metrics['evaluation']['regression'][protocol]
        summary_rows.append(f'<tr><td>{protocol}</td><td>{c["Environment logistic"]["accuracy"]:.1%} → {c["Environment logistic + age/gender"]["accuracy"]:.1%}</td><td>{c["Environment logistic"]["balanced_accuracy"]:.1%} → {c["Environment logistic + age/gender"]["balanced_accuracy"]:.1%}</td><td>{r["Environment ridge"]["mae"]:.3f} → {r["Environment ridge + age/gender"]["mae"]:.3f}</td></tr>')
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Step 1: age and gender</title><style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#253243}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}h2{{margin-top:32px}}code{{overflow-wrap:anywhere}}</style></head><body><h1>Step 1: add age and recorded gender</h1><p>Both valence classification and regression; November 10, 2026. This step adds demographics only. Thermal-background questions and study subject are not predictors yet.</p>
    <h2>Matched comparison</h2><p>Same 163 responses, 22 students and nine dates; same seven environmental inputs; same excluded conflicting labels, fixed hyperparameters and evaluation splits. Added Age and one binary indicator for recorded Female (Male is the reference). Both fields are complete. No student identifier enters a model. Age distribution among the 22 students: {html.escape(str(m['student_age_counts']))}; gender distribution: {html.escape(str(m['student_gender_counts']))}. These counts describe students, not repeated survey rows.</p><table><tr><th>Evaluation</th><th>Classification accuracy: environment → + demographics</th><th>Balanced accuracy: environment → + demographics</th><th>Regression MAE: environment → + demographics</th></tr>{''.join(summary_rows)}</table>
    <p>The linear classification comparison uses logistic regression (C=1), and regression uses ridge (alpha=1), with training-only imputation/scaling. Forest comparisons retain the previous 300-tree, depth-3, minimum-leaf-10 settings. All environments-only held-out predictions were reproduced and verified against the previous saved predictions. Classification predicts valence 1–2 versus 3–4; regression predicts the original 1–4 score and clips predictions to those bounds.</p>
    <p>Held-out date keeps whole dates out, but students can appear on other training dates. Future block trains on the October dates and tests on four later November dates. Held-out student keeps each student out but shares dates, so it measures a different kind of generalization. Improvement on dates alone can reflect stable student differences; it does not establish a demographic effect or transfer to unseen students.</p>'''
    for protocol in ['held_out_date','future_block','held_out_student']:
        text+=f'<h2>{protocol}: classification</h2>'+table('classification',protocol)+f'<h2>{protocol}: regression</h2>'+table('regression',protocol)
    text+='''<h2>Uncertainty</h2><p>metrics.json includes paired changes for linear and forest models. Date-based evaluations resample dates; student-held-out evaluation resamples students, using the fixed held-out predictions without refitting. There are only nine or four date clusters and 22 student clusters, so these intervals do not represent full training uncertainty. These are sequential exploratory comparisons after seeing earlier models, not an untouched confirmatory experiment.</p>'''
    text+=f'''<h2>Do we know what students are studying?</h2><p>Yes: timetable subjects include {', '.join(subjects['all_timetable_subjects'])}. Class-group membership can identify the eligible scheduled classes for a student without knowing the environmental filename crosswalk. It does not prove attendance or actual subject activity at survey submission.</p><p>For the same 163 mood observations, {counts.get('current_scheduled_class',0)} have one eligible class scheduled at the submission time. Another {counts.get('recent_class_proxy_only',0)} can be associated only with a class that ended within the preceding 30 minutes, and must be labeled as a recent-class proxy. Other observations fall before classes, on days without eligible timetable entries (including Wednesdays), in long gaps, or in ambiguous schedules. Exact counts and per-response audit statuses are saved in subject_availability.csv. The direct current-class subject counts are {html.escape(str(subjects['current_class_subjects']))}.</p><p>We have not added subject, subject inferred from mood, or a forced recent/current class label to this step. Mood describes the present moment, whereas engagement questions refer to the previous class. The timetable Date/Week inconsistencies previously identified in class IDs 245 and 248 remain unresolved; this audit uses the provided Week/Weekday and excludes neither inconsistency by silently rewriting it.</p>
    <h2>Outputs and next step</h2><p><a href="metrics.json">Metrics, paired changes and metadata</a>; <a href="classification_predictions.csv">classification held-out predictions</a>; <a href="regression_predictions.csv">regression held-out predictions</a>; <a href="subject_availability.csv">subject availability audit</a>. Fitted comparison models and enriched research observations are saved locally. Prior classification/regression artifacts are unchanged. Stop here for review before adding thermal-background responses or a subject feature.</p><p>Source files: {html.escape(m['student_file'])}; prepared modeling observations from the previous environmental model; the verified local timetable and survey source rows. Source hashes, feature names and software-independent audit details appear in metrics.json. Run demographics_step1.py to reproduce this step.</p></body></html>'''
    (DEST/'report.html').write_text(text)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/broad-environment-model/modeling_data.csv';df=pd.read_csv(source,parse_dates=['timestamp','feature_end'])
    meta=json.loads((ROOT/'outputs/broad-environment-model/metrics.json').read_text())['metadata'];data_dir=Path(meta['input_dir'])
    student_path=Path('/Users/nishatalagala/Downloads/student.csv');students=pd.read_csv(student_path)
    assert students.Pid.is_unique
    assert students.equals(pd.read_csv(data_dir/'participant_class_info/student.csv'))
    order=df.source_row.to_numpy().copy()
    df=df.merge(students[['Pid','Age','Gender']],on='Pid',how='left',validate='many_to_one',sort=False)
    assert np.array_equal(order,df.source_row) and len(df)==163 and df[['Age','Gender']].notna().all().all()
    assert df.Gender.isin(['Female','Male']).all()
    df['recorded_female']=df.Gender.eq('Female').astype(int)
    audit,audit_summary=subject_audit(df,students,data_dir);audit.to_csv(DEST/'subject_availability.csv',index=False)
    class_specs=extend_specs(classification.specifications(),['Majority baseline','Time of day logistic'],'Environment logistic')
    reg_specs=extend_specs(regression.specifications(),['Mean baseline','Median baseline','Time of day ridge'],'Environment ridge')
    results={'classification':{},'regression':{}};predictions={'classification':[],'regression':[]}
    for protocol in ['held_out_date','future_block','held_out_student']:
        results['classification'][protocol],p=classification.run_protocol(df,class_specs,protocol);predictions['classification'].append(p)
        results['regression'][protocol],p=regression.evaluate(df,reg_specs,protocol);predictions['regression'].append(p)
        print('Completed',protocol,flush=True)
    predictions={k:pd.concat(v,ignore_index=True) for k,v in predictions.items()}
    for task,p in predictions.items():compare_saved_predictions(p,task);p.to_csv(DEST/(task+'_predictions.csv'),index=False)
    intervals={}
    for task,linear in [('classification','Environment logistic'),('regression','Environment ridge')]:
        intervals[task]={protocol:{name:paired_interval(predictions[task],task,protocol,name+' + age/gender',name) for name in [linear,'Environment forest']} for protocol in ['held_out_date','future_block','held_out_student']}
    enriched=df.drop(columns=['Clothing','Engage_1','Engage_2','Engage_3','Engage_4','Engage_5','Thermal_preference','Thermal_sensation','Loc_x','Loc_y','Confidence_level'],errors='ignore')
    enriched.to_csv(DEST/'modeling_data.csv',index=False)
    present=students[students.Pid.isin(df.Pid)]
    metadata={'step':'Age and recorded gender only','n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'student_age_counts':present.Age.value_counts().sort_index().to_dict(),'student_gender_counts':present.Gender.value_counts().to_dict(),'student_file':str(student_path),'student_sha256':hashlib.sha256(student_path.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'added_features':DEMOGRAPHICS,'subject_used_as_feature':False,'thermal_background_used_as_feature':False,'baseline_predictions_match_previous_runs':True}
    for task,specs,target in [('classification',class_specs,'higher_valence'),('regression',reg_specs,'Valence')]:
        fitted={name:clone(model).fit(df[cols],df[target]) for name,(cols,model) in specs.items()}
        payload={'models':fitted,'features':{name:cols for name,(cols,_) in specs.items()},'target':target,'bounds':(1,4) if task=='regression' else None,'gender_encoding':{'Female':1,'Male':0},'metadata':metadata}
        joblib.dump(payload,DEST/(task+'_models.joblib'));restored=joblib.load(DEST/(task+'_models.joblib'))
        name=('Environment logistic' if task=='classification' else 'Environment ridge')+' + age/gender';cols=specs[name][0]
        np.testing.assert_allclose(restored['models'][name].predict(df[cols]),fitted[name].predict(df[cols]))
    output={'metadata':metadata,'evaluation':results,'paired_changes':intervals,'subject_audit':audit_summary};(DEST/'metrics.json').write_text(json.dumps(output,indent=2));make_report(output)
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
