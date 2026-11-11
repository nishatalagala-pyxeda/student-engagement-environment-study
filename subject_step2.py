"""Step 2: scheduled subject/context, added to the same demographic models."""
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
import broad_environment_model as classification
import valence_regression as regression
import demographics_step1 as previous

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/subject-step2'
SUBJECT='scheduled_subject_context'

def add_subject_pipeline(template,numeric):
    estimator=clone(template.steps[-1][1])
    is_forest=estimator.__class__.__name__.startswith('RandomForest')
    numeric_pipe=SimpleImputer(strategy='median') if is_forest else make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    prep=ColumnTransformer([('numeric',numeric_pipe,numeric),('subject',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[SUBJECT])],remainder='drop')
    return make_pipeline(prep,estimator)

def specs_for(task):
    if task=='classification':
        specs=previous.extend_specs(classification.specifications(),['Majority baseline','Time of day logistic'],'Environment logistic')
        linear='Environment logistic + age/gender';baselines=['Majority baseline','Time of day logistic']
    else:
        specs=previous.extend_specs(regression.specifications(),['Mean baseline','Median baseline','Time of day ridge'],'Environment ridge')
        linear='Environment ridge + age/gender';baselines=['Mean baseline','Median baseline','Time of day ridge']
    names=baselines+[linear,'Environment forest + age/gender','Age/gender only']
    result={name:specs[name] for name in names};pairs=[]
    for name in [linear,'Environment forest + age/gender','Age/gender only']:
        cols,model=specs[name];new='Age/gender + subject' if name=='Age/gender only' else name+' + subject'
        result[new]=(cols+[SUBJECT],add_subject_pipeline(model,cols));pairs.append((name,new))
    assert all(not set(cols).intersection({'Valence','higher_valence','Arousal','Pid','source_row'}) for cols,_ in result.values())
    return result,pairs

def validate_old_predictions(pred,task):
    prior=pd.read_csv(ROOT/'outputs/demographics-step1'/(task+'_predictions.csv'))
    common=set(pred.model).intersection(prior.model)
    value='probability_higher' if task=='classification' else 'predicted_valence'
    for name in common:
        a=pred[pred.model.eq(name)].set_index(['protocol','source_row']).sort_index()
        b=prior[prior.model.eq(name)].set_index(['protocol','source_row']).sort_index()
        assert a.index.equals(b.index)
        np.testing.assert_allclose(a[value],b[value],rtol=0,atol=1e-9)

def make_report(output):
    def table(task,protocol):
        cols=['accuracy','balanced_accuracy','macro_f1'] if task=='classification' else ['mae','rmse','r2']
        headers=['Accuracy ↑','Balanced accuracy ↑','Macro F1 ↑'] if task=='classification' else ['MAE ↓','RMSE ↓','R² ↑']
        rows=''.join('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[c]:.3f}</td>' for c in cols)+'</tr>' for name,v in output['evaluation'][task][protocol].items())
        return '<table><tr><th>Model</th>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'+rows+'</table>'
    c=output['evaluation']['classification']['held_out_date'];r=output['evaluation']['regression']['held_out_date'];counts=output['metadata']['subject_context_counts']
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Step 2: subject</title><style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#253243}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#edf2f7}}h2{{margin-top:32px}}code{{overflow-wrap:anywhere}}</style></head><body>
    <h1>Step 2: add scheduled subject/context</h1><p>Classification and regression of reported valence; November 10, 2026.</p>
    <p>For the linear combined model, adding subject/context changes held-out-date classification accuracy from {c['Environment logistic + age/gender']['accuracy']:.1%} to {c['Environment logistic + age/gender + subject']['accuracy']:.1%}, balanced accuracy from {c['Environment logistic + age/gender']['balanced_accuracy']:.1%} to {c['Environment logistic + age/gender + subject']['balanced_accuracy']:.1%}, and ridge regression MAE from {r['Environment ridge + age/gender']['mae']:.3f} to {r['Environment ridge + age/gender + subject']['mae']:.3f}. See baselines and generalization checks below before interpreting a gain.</p>
    <h2>Feature definition and availability</h2><p>The same 163 observations from 22 students on nine dates are retained. There are 121 unique current scheduled-class matches. Subjects are based only on timetable Week/Weekday, submission time and the student's form/maths/language groups. They are not inferred from reported mood. Remaining observations retain explicit schedule-context categories; a recently finished class is not relabeled as the current subject.</p>
    <table><tr><th>Scheduled subject/context</th><th>Responses</th></tr>{''.join('<tr><td>'+html.escape(k)+'</td><td>'+str(v)+'</td></tr>' for k,v in counts.items())}</table>
    <p>Outside scheduled class combines the 22 recent-class-proxy cases and one longer gap. No timetable entry covers 17 cases; before scheduled classes covers two. A matching class describes scheduled activity, not observed attendance or current room. The known timetable inconsistencies and absent Wednesday entries remain. We do not rewrite dates or force a subject assignment.</p>
    <h2>Controlled comparison</h2><p>Compare each age/gender model against itself plus one categorical subject/context field. Its categories are one-hot encoded inside each training fold, not assigned numeric ranks. Unseen test categories are encoded with no active subject indicator. Existing numerical preprocessing and all estimator hyperparameters are retained; there is no tuning or outcome-based selection. Exact historical predictions for unchanged models were reproduced and checked.</p>
    <p>Environmental inputs remain the seven previous broad proxies. Age and recorded Female/Male indicator remain included. No student ID, arousal, thermal-background answer or other survey response is a predictor. Classification retains the 1–2 versus 3–4 valence split; regression predicts the original 1–4 scale and applies fixed bounds.</p>
    <p>Main evaluation holds out whole dates; the chronological check trains on 105 October responses and tests on 58 later-November responses. Supplemental student-held-out evaluation shares dates and cannot substitute for unseen-context evaluation. Preprocessing is fitted only on training folds. Category availability diagnostics are in metrics.json.</p>'''
    for protocol in ['held_out_date','future_block','held_out_student']:
        text+=f'<h2>{protocol}: classification</h2>'+table('classification',protocol)+f'<h2>{protocol}: regression</h2>'+table('regression',protocol)
    text+='''<h2>Interpretation and uncertainty</h2><p>Subject/context may describe schedule and time patterns rather than the effect of learning a particular subject. Classes, classmates, teachers and dates are confounded, and campus environmental inputs are not measured individual classroom exposures. Stepwise experimentation has already inspected earlier results, so these are exploratory comparisons, not an untouched final validation.</p><p>Paired-change intervals in metrics.json resample dates (nine or four clusters) or students (22 clusters) from fixed held-out predictions, without retraining. They do not account for all model uncertainty or both crossed student/date dependencies. Small subject categories such as Form have particularly limited support. No causal subject or demographic effects are claimed.</p><h2>Outputs</h2><p><a href="metrics.json">Metrics and paired comparisons</a>; <a href="classification_predictions.csv">classification predictions</a>; <a href="regression_predictions.csv">regression predictions</a>; <a href="modeling_data.csv">prepared observations with subject/context</a>. Separate fitted model payloads store required features and the trained categorical encoders. Sources and SHA256 values are recorded in metrics.json. Run subject_step2.py to reproduce the step. Regression predictions must be clipped to the stored 1–4 bounds. Earlier artifacts are unchanged. Thermal-background predictors have not yet been added.</p></body></html>'''
    (DEST/'report.html').write_text(text)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/demographics-step1/modeling_data.csv';audit_path=ROOT/'outputs/demographics-step1/subject_availability.csv'
    df=pd.read_csv(source,parse_dates=['timestamp','feature_end']);audit=pd.read_csv(audit_path)
    assert df.source_row.is_unique and audit.source_row.is_unique
    before=df.source_row.to_numpy().copy()
    df=df.merge(audit[['source_row','participant','date','status','scheduled_subject_or_recent_proxy']],on=['source_row','date'],validate='one_to_one',how='left',sort=False)
    assert np.array_equal(before,df.source_row) and df.Pid.eq(df.participant).all()
    df[SUBJECT]='Unknown schedule context'
    current=df.status.eq('current_scheduled_class');assert current.sum()==121
    df.loc[current,SUBJECT]=df.loc[current,'scheduled_subject_or_recent_proxy']
    df.loc[df.status.isin(['recent_class_proxy_only','outside_class_or_long_gap']),SUBJECT]='Outside scheduled class'
    df.loc[df.status.eq('no_eligible_timetable_for_day'),SUBJECT]='No timetable entry'
    df.loc[df.status.eq('before_first_scheduled_class'),SUBJECT]='Before scheduled classes'
    assert df[SUBJECT].notna().all() and df[SUBJECT].ne('Unknown schedule context').all() and len(df)==163
    results={};allpred={};intervals={};fitted_payloads={};category_checks=[]
    for task in ['classification','regression']:
        specs,pairs=specs_for(task);results[task]={};preds=[]
        for protocol in ['held_out_date','future_block','held_out_student']:
            scores,p=(classification.run_protocol if task=='classification' else regression.evaluate)(df,specs,protocol);results[task][protocol]=scores;preds.append(p)
            print('Completed',task,protocol,flush=True)
        pred=pd.concat(preds,ignore_index=True);validate_old_predictions(pred,task);pred.to_csv(DEST/(task+'_predictions.csv'),index=False);allpred[task]=pred
        intervals[task]={}
        for protocol in ['held_out_date','future_block','held_out_student']:
            intervals[task][protocol]={}
            for old,new in pairs:
                v=previous.paired_interval(pred,task,protocol,new,old);v['unit']=v['unit'].replace('demographics','subject addition');intervals[task][protocol][new]=v
        target='higher_valence' if task=='classification' else 'Valence'
        fitted={name:clone(model).fit(df[cols],df[target]) for name,(cols,model) in specs.items()}
        fitted_payloads[task]={'models':fitted,'features':{name:cols for name,(cols,_) in specs.items()},'target':target,'bounds':(1,4) if task=='regression' else None,'gender_encoding':{'Female':1,'Male':0},'subject_encoding':'training-only one-hot; unknown categories ignored'}
    # Only diagnose category coverage; do not fit categories using test data.
    for date in sorted(df.date.unique()):
        train=set(df.loc[df.date.ne(date),SUBJECT]);test=df[df.date.eq(date)]
        unknown=test[~test[SUBJECT].isin(train)]
        category_checks.append({'held_out_date':date,'unseen_categories':sorted(set(unknown[SUBJECT])),'affected_rows':len(unknown)})
    train_categories=set(df.loc[df.timestamp.lt('2019-11-02'),SUBJECT]);future=df[df.timestamp.ge('2019-11-18')]
    future_unknown=future[~future[SUBJECT].isin(train_categories)]
    metadata={'step':'Add scheduled subject/context; retain demographics','n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'added_categorical_feature':SUBJECT,'subject_context_counts':df[SUBJECT].value_counts().to_dict(),'old_predictions_reproduced':True,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'thermal_background_used':False}
    output={'metadata':metadata,'evaluation':results,'paired_changes':intervals,'category_checks':category_checks,'future_unseen_categories':{'categories':sorted(set(future_unknown[SUBJECT])),'affected_rows':len(future_unknown)}}
    for task,payload in fitted_payloads.items():
        payload['metadata']=metadata;joblib.dump(payload,DEST/(task+'_models.joblib'));restored=joblib.load(DEST/(task+'_models.joblib'))
        name=('Environment logistic' if task=='classification' else 'Environment ridge')+' + age/gender + subject';cols=payload['features'][name]
        np.testing.assert_allclose(restored['models'][name].predict(df[cols]),payload['models'][name].predict(df[cols]))
    df.to_csv(DEST/'modeling_data.csv',index=False);(DEST/'metrics.json').write_text(json.dumps(output,indent=2));make_report(output);print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':main()
