"""Paper-defined engagement dimensions; context/sensor prediction without engagement-answer leakage."""
from pathlib import Path
import os,json,hashlib,zipfile,sys,platform,html
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
import sklearn,scipy,joblib
from sklearn.base import clone
import all_predictors as tab
import thermal_models as common
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/engagement-models'
ITEMS=[f'Engage_{i}' for i in range(1,6)]
TARGETS=[('overall_classification','overall_class',False),('overall_regression','overall_engagement',True),('behavioral_regression','behavioral_engagement',True),('emotional_regression','emotional_engagement',True),('cognitive_regression','cognitive_engagement',True)]


def prepare():
    data_dir=Path(json.loads((ROOT/'outputs/broad-environment-model/metrics.json').read_text())['metadata']['input_dir'])
    raw=pd.read_csv(data_dir/'survey/student_survey.csv')
    with zipfile.ZipFile('/Users/nishatalagala/Downloads/Figshare-Navya/Survey.zip') as z:original=pd.read_csv(z.open('Survey/Student_survey.csv'))
    cols=[c for c in original if c!='Time'];assert raw[cols].fillna('__NA__').equals(original[cols].fillna('__NA__'))
    raw['source_row']=raw.index+2;raw['timestamp']=pd.to_datetime(original.Time,dayfirst=True)
    assert raw[ITEMS].notna().all().all()
    for c in ITEMS:assert raw[c].isin(tab.LIKERT).all()
    # Clean target conflicts before selecting environmental availability, independently of mood/comfort.
    conflicts=raw.groupby(['Pid','timestamp'])[ITEMS].transform('nunique').gt(1).any(axis=1)
    clean=raw[~conflicts].drop_duplicates(['Pid','timestamp']).copy()
    prepared=pd.read_csv(ROOT/'outputs/thermal-models/modeling_data.csv',parse_dates=['timestamp','feature_end'])
    features=json.loads((ROOT/'outputs/thermal-models/scope_features.json').read_text())
    scoped={}
    cats={'scheduled_subject_context','form_group','math_group','language_group','study_week','scheduled_room','recent_subject'}
    for name,cols in features.items():scoped[name]=([c for c in cols if c not in cats],[c for c in cols if c in cats])
    # The cached environmental cohort covers every nonconflicting later engagement response.
    eligible=clean[clean.timestamp.ge('2019-10-07')]
    assert set(eligible.source_row).issubset(set(prepared.source_row)), 'Uncached eligible response: prepare its environmental/sensor fields instead of dropping it.'
    df=prepared[prepared.source_row.isin(clean.source_row)].copy().reset_index(drop=True)
    assert set(df.source_row)==set(eligible.source_row)
    source=clean.set_index('source_row').loc[df.source_row]
    assert np.array_equal(source.Pid,df.Pid)
    assert np.array_equal(source.timestamp,df.timestamp)
    for c in ITEMS:assert np.array_equal(source[c],df[c])
    coded=np.column_stack([df[c].map(tab.LIKERT).to_numpy(float)*(-1 if i in [2,4] else 1) for i,c in enumerate(ITEMS,1)])
    df['behavioral_engagement']=coded[:,:2].mean(axis=1)+3
    df['emotional_engagement']=coded[:,2:4].mean(axis=1)+3
    df['cognitive_engagement']=coded[:,4]+3
    df['overall_engagement']=coded.mean(axis=1)+3
    df['overall_class']=np.where(df.overall_engagement.gt(3),'Higher than neutral','Neutral or lower')
    for _,target,regr in TARGETS:
        if regr:assert df[target].between(1,5).all()
    forbidden=set(ITEMS+['overall_class','overall_engagement','behavioral_engagement','emotional_engagement','cognitive_engagement','Valence','Arousal','Pid','source_row','date','Thermal_sensation','Thermal_preference','Confidence_level'])
    for nums,category in scoped.values():
        assert not set(nums+category)&forbidden
        assert not any(c.startswith('engagement_item_') for c in nums+category)
        assert len(nums+category)==len(set(nums+category))
    assert not df.duplicated(['Pid','timestamp']).any()
    # Audits preserve conflicts across all survey dates, not just the modeled subset.
    raw[conflicts].to_csv(DEST/'excluded_conflicting_engagement.csv',index=False)
    clean[~clean.source_row.isin(df.source_row)].to_csv(DEST/'excluded_environment_unavailable.csv',index=False)
    metadata={'raw_responses':len(raw),'conflicting_engagement_rows':int(conflicts.sum()),'redundant_same_time_rows_removed':int((~conflicts).sum()-len(clean)),'clean_engagement_responses':len(clean),'excluded_no_environment':len(clean)-len(df),'n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'date_counts':df.date.value_counts().sort_index().to_dict(),'class_counts':df.overall_class.value_counts().to_dict(),'score_summary':df[[target for _,target,regr in TARGETS if regr]].describe().to_dict(),'primary_models':{'overall_classification':'Context | forest','regression':'Context | forest'},'score_definition':'Code agreement -2..2; reverse items 2 and 4; behavioral=mean(1,2 reversed), emotional=mean(3,4 reversed), cognitive=item5, overall=mean(all five); add3 to place scores on1..5. Overall is item-weighted, not equal-weight mean of dimensions.','classification_definition':'Higher than neutral: overall score >3; Neutral or lower: score <=3. Exploratory fixed threshold, not a validated clinical/educational cutoff.','timing_limit':'Questions ask about the last class. Features summarize at most30 minutes strictly before survey; these need not coincide with that class. Scheduled subject at submission is not verified target-class attendance.','paper_sources':['/Users/nishatalagala/Downloads/s41597-022-01347-w.pdf','/Users/nishatalagala/Downloads/3411813.pdf'],'cache_input_sha256':hashlib.sha256((ROOT/'outputs/thermal-models/modeling_data.csv').read_bytes()).hexdigest(),'source_survey_sha256':hashlib.sha256((data_dir/'survey/student_survey.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'shared_evaluator_sha256':hashlib.sha256((ROOT/'thermal_models.py').read_bytes()).hexdigest(),'feature_preparation':'Reuses verified thermal context/wearable features, while cleaning independently on all five engagement items. No selection by mood or comfort labels. All eligible later engagement responses confirmed cached.','all_engagement_items_excluded_as_predictors':True,'scope_features':features}
    return df,scoped,metadata


def report(output):
    m=output['metadata'];ev=output['evaluation'];text='''<!doctype html><html><head><meta charset="utf-8"><title>Engagement prediction</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}</style></head><body><h1>Predicting student engagement</h1><p>Exploratory follow-on analysis; November 10, 2026.</p>'''
    text+=f'''<p>Started with {m['raw_responses']} survey responses. Excluded {m['conflicting_engagement_rows']} rows with conflicting engagement items for the same student/time, collapsed {m['redundant_same_time_rows_removed']} redundant responses, and excluded {m['excluded_no_environment']} clean September responses without environmental coverage. All comparisons use <strong>{m['n']} responses, {m['students']} students, {m['dates']} dates</strong>, including surveys without mood labels. Cleaning is independent of comfort and mood outcomes.</p>
    <h2>Paper-defined scores and prediction targets</h2><p>The source data paper and n-Gage paper specify reversing items2 and4. Behavioral engagement averages items1 and2 reversed, emotional averages3 and4 reversed, cognitive is item5, and overall averages all five coded items. Scores are shifted to1–5, higher meaning more reported engagement. Overall weights each item equally; it is not the equal-weight average of the three dimensions. Regression treats score intervals as approximately equal. Papers: <a href="/Users/nishatalagala/Downloads/s41597-022-01347-w.pdf">dataset description</a> and <a href="/Users/nishatalagala/Downloads/3411813.pdf">n-Gage (Table4 and Section6)</a>.</p><p>Primary targets: overall score regression and overall classification (higher than neutral, score&gt;3, versus neutral or lower, score≤3). The fixed classification threshold is exploratory, not a validated cutoff. Dimension regressions are secondary; cognitive is only one item. No engagement item or derived score is an input to any model.</p>
    <h2>Predictors and evaluation</h2><p>Same fixed models and preprocessing as the earlier thermal/valence comparisons. Environment:17 trailing outdoor/campus variables; context adds background demographics/thermal questions, groups, subject and schedule/time. Comparisons add clothing/seating and/or44 pre-survey wearable signal/quality summaries. Current comfort, mood/arousal, confidence, participant/session/date identifiers and all engagement answers are excluded. No target-derived feature selection or tuning. See <a href="scope_features.json">exact feature lists</a>.</p><p>Leave one complete date out; train October→test later November; and leave one student out. All encoders/imputers/scalers are fitted inside the training fold. Logistic C=1; ridge alpha=1; forest300 trees, depth3 and minimum leaf10, with fixed seed42 and the same feature sampling as earlier runs. Regression predictions are clipped to1–5. Context forest is the primary model; other scopes/families are exploratory comparisons. Majority, mean and median baselines are fitted in every training fold. This is not a reproduction of n-Gage's class-window/synchrony features, LightGBM tuning or cross-validation.</p>'''
    c=ev['overall_classification'];r=ev['overall_regression'];name='Context | forest'
    text+=f'''<h2>Primary results</h2><p>Overall classification accuracy on held-out dates: {c['held_out_date'][name]['accuracy']:.1%}, versus {c['held_out_date']['Majority baseline']['accuracy']:.1%} for the majority baseline. Later-November accuracy: {c['future_block'][name]['accuracy']:.1%}, versus {c['future_block']['Majority baseline']['accuracy']:.1%}. Overall regression MAE on held-out dates: {r['held_out_date'][name]['mae']:.3f}, versus {r['held_out_date']['Median baseline']['mae']:.3f} for the median baseline. Later-November MAE: {r['future_block'][name]['mae']:.3f}, versus {r['future_block']['Median baseline']['mae']:.3f}. Lower regression error is better. Read all protocols, balanced accuracy and uncertainty intervals before interpreting a gain.</p>'''
    for target,protocols in ev.items():
        text+='<h2>'+html.escape(target)+'</h2>'
        for protocol,models in protocols.items():
            keys=['accuracy','balanced_accuracy','macro_f1'] if target.endswith('classification') else ['mae','rmse','r2']
            text+='<h3>'+protocol+'</h3><table><tr><th>Model</th>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
            for name,v in models.items():text+='<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[k]:.3f}</td>' for k in keys)+'</tr>'
            text+='</table>'
    text+='''<h2>What this can establish</h2><p>The questionnaires ask about the last class, while sensors summarize the30 minutes before submission. Delayed responses and changes of class can misalign features and outcomes. Classroom sensor mapping remains unknown; indoor fields are campus proxies, not personal room exposure. Date holdouts retain known students; student holdouts retain shared environmental dates. Nine dates are few independent contexts. These follow earlier experiments and are exploratory; selecting the best score now would introduce further selection bias. Wearable availability, imperfect clock alignment, off-wrist/motion artifacts and timetable uncertainty remain. Classification thresholds and single/short-item subscales limit interpretation. No causal, clinical or deployment claim follows.</p><p>Paired intervals resample fixed held-out date/student prediction clusters without refitting; they do not capture full crossed-dependency, training or model-selection uncertainty. <a href="metrics.json">Metrics, counts, confusion matrices and paired intervals</a>; <a href="modeling_data.csv">prepared observations</a>; <a href="scope_features.json">predictor lists</a>; <a href="integrity_checks.json">integrity checks</a>; <a href="runtime.json">runtime versions</a>. Target-specific held-out predictions and fitted pipelines are saved beside this report. Original source data and prior model results remain unchanged. Run engagement_models.py to reproduce.</p></body></html>'''
    (DEST/'report.html').write_text(text)


def main():
    DEST.mkdir(parents=True,exist_ok=True);df,scopes,metadata=prepare();df.to_csv(DEST/'modeling_data.csv',index=False)
    (DEST/'scope_features.json').write_text(json.dumps(metadata['scope_features'],indent=2))
    (DEST/'runtime.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{m.__name__:m.__version__ for m in [sklearn,scipy,np,pd,joblib]},'pythonpath':os.environ.get('PYTHONPATH','')},indent=2))
    (DEST/'integrity_checks.json').write_text(json.dumps({'original_released_survey_values_verified':True,'target_cleaning_independent_of_comfort_and_mood':True,'all_eligible_later_responses_cached':True,'no_engagement_answer_in_any_predictor_scope':True,'same_participant_timestamp_source_items_verified':True,'score_bounds_1_to_5_verified':True,'no_duplicate_student_timestamps':True},indent=2))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['scope_features','score_summary']},indent=2),flush=True)
    evaluation={};intervals={}
    for name,target,regr in TARGETS:
        models=common.specs(scopes,regr);evaluation[name]={};parts=[]
        for protocol in common.PROTOCOLS:
            score,pred=common.evaluate(df,models,target,regr,protocol,bounds=(1,5));evaluation[name][protocol]=score;parts.append(pred);print('Completed',name,protocol,flush=True)
        pred=pd.concat(parts,ignore_index=True);pred.to_csv(DEST/(name+'_predictions.csv'),index=False);intervals[name]=common.paired(pred,target,regr)
        fitted={n:clone(template).fit(df[cols],df[target]) for n,(cols,template) in models.items()}
        payload={'models':fitted,'features':{n:cols for n,(cols,_) in models.items()},'bounds':(1,5) if regr else None,'target':target,'metadata':metadata}
        file=DEST/(name+'_models.joblib');joblib.dump(payload,file);restored=joblib.load(file);cols=payload['features']['Context | forest']
        np.testing.assert_array_equal(restored['models']['Context | forest'].predict(df[cols]),fitted['Context | forest'].predict(df[cols]))
    output={'metadata':metadata,'evaluation':evaluation,'paired_against_baseline':intervals}
    (DEST/'metrics.json').write_text(json.dumps(output,indent=2));report(output);print('All engagement models complete.',flush=True)

if __name__=='__main__':main()
