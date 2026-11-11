"""Fixed prior OSE algorithms evaluated on entirely held-out student groups."""
from pathlib import Path
import hashlib, html, json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import ose_all_models as previous

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/ose-all-variables'
OUT=ROOT/'outputs/ose-unseen-students'
PLAN={
 'question':'Predict supplied OSE engagement for students excluded completely from model training, without any past OSE input.',
 'models':'Unchanged Ridge alpha=10 and random forest 300 trees, depth=3, minimum leaf=10, seed=42. No tuning or new algorithms.',
 'split':'Five shuffled GroupKFold folds, random_state=42. Group on numeric ID suffix across cohorts conservatively: cross-cohort physical identity is unverified. All records from each resulting group remain in one fold. No identifiers as predictors.',
 'cohorts':'Primary all 1530 cleaned reports, including no prior OSE or stress. Secondary original 1118 historically eligible records for comparison, but student splits differ from previous chronological test.',
 'features':'All previously verified derived predictor groups, excluding every ose_* field and target/identity/eligibility fields. Historical non-OSE observations from a test student are allowed as inputs, never training labels. Reuse per-target features strictly preceding each response with original buffers. No previous OSE required for primary inclusion.',
 'preprocessing':'Fit imputation, scaling, categorical encoding and facial PCA only on training students separately within every fold. Test subjects never contribute to these fitted transformations.',
 'baseline':'Fold training population mean and median. Main MAE weights reports equally; also report equal-student MAE. Bootstrap differences by held-out student group, 3000 draws, fixed predictions.',
 'time':'This tests transfer across students in the observed study period; training and testing dates can overlap. It does not establish future-calendar forecasting for new students.',
 'limits':'Cohort identifiers may not uniquely identify physical people. Grouping identical numeric suffixes across cohorts is conservative and may merge different people. Historical non-OSE measurements must be available at deployment; not zero-information arrival. Static demographic administration dates unknown. Supplied score coding unverified. Reused exploratory dataset, no independent external validation. Existing modality/source quality limits remain.'
}

def load():
    schema=json.loads((SOURCE/'feature_schema.json').read_text())
    d=pd.read_csv(SOURCE/'modeling_features.csv',float_precision='round_trip',dtype={c:str for c in schema['categorical_features']})
    d['date']=pd.to_datetime(d.date)
    d['student_group']=d.person.str.extract(r'(\d+)$',expand=False)
    assert d.student_group.notna().all()
    cols=previous.scopes(schema)['All without past OSE']
    assert not any(c.startswith('ose_') for c in cols)
    assert not set(cols)&{'OSE','eligible','source_row','person','student_group','date','ID'}
    return schema,d,cols

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    schema,d,cols=load()
    plan={**PLAN,'input_features':len(cols),'source_hashes':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['modeling_features.csv','feature_schema.json']}}
    (OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    predictions=[];members=[];folds=[];models={}
    for protocol,frame in [('All students',d),('Previous eligible subset',d[d.eligible])]:
        frame=frame.reset_index(drop=True)
        for fold,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(frame,groups=frame.student_group),1):
            train=frame.iloc[tr];test=frame.iloc[te]
            assert set(train.student_group).isdisjoint(test.student_group)
            assert set(train.person).isdisjoint(test.person)
            folds.append({'protocol':protocol,'fold':fold,'train_reports':len(train),'test_reports':len(test),'train_ids':train.person.nunique(),'test_ids':test.person.nunique(),'train_groups':train.student_group.nunique(),'test_groups':test.student_group.nunique()})
            for role,data in [('train',train),('test',test)]:
                members.extend(data[['source_row','person','student_group','date','OSE']].assign(protocol=protocol,fold=fold,role=role).to_dict('records'))
            preds={'Population mean':np.repeat(train.OSE.mean(),len(test)),'Population median':np.repeat(train.OSE.median(),len(test))}
            for forest,name in [(False,'Ridge'),(True,'Random forest')]:
                model=previous.build(cols,schema,forest).fit(train[cols],train.OSE)
                preds[name]=model.predict(test[cols]);models[(protocol,fold,name)]=model
            for name,values in preds.items():
                assert np.isfinite(values).all()
                predictions.extend({'protocol':protocol,'fold':fold,'model':name,'source_row':int(r.source_row),'person':r.person,'student_group':r.student_group,'date':str(r.date.date()),'observed':float(r.OSE),'predicted':float(v)} for (_,r),v in zip(test.iterrows(),values))
            print(protocol,'fold',fold,'train',len(train),'test',len(test),flush=True)
    p=pd.DataFrame(predictions);p.to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame(members).to_csv(OUT/'fold_membership.csv',index=False)
    joblib.dump({'models':models,'features':cols,'schema':schema,'plan':plan},OUT/'fold_models.joblib')
    scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={}
        for name,g in pp.groupby('model'):
            errors=abs(g.predicted-g.observed)
            scores[protocol][name]={**previous.core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':g.person.nunique(),'groups':g.student_group.nunique(),'equal_student_mae':float(errors.groupby(g.person).mean().mean())}
        for name in ['Ridge','Random forest']:
            for base in ['Population mean','Population median']:
                a=pp[pp.model.eq(name)].copy();b=pp[pp.model.eq(base)].copy()
                # Resample conservative physical-identity groups, not separate cohort labels.
                a['person']=a.student_group;b['person']=b.student_group
                comparisons[protocol][name+' versus '+base]=previous.paired(a,b)
    result={'plan':plan,'scores':scores,'comparisons':comparisons,'folds':folds}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=int))
    render(result)
    print(json.dumps(scores,indent=2,default=int),flush=True)

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>OSE for unseen students</title><style>body{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2f7}</style><h1>Predicting OSE for unseen students</h1><p>All prior predictor groups, excluding past OSE. Fixed previous ridge and random forest algorithms. Five student-held-out folds: each student appears in testing once, with no records in that fold’s training. OSE denotes online student engagement.</p>'
    s=r['scores']['All students'];gain=r['comparisons']['All students']['Random forest versus Population mean']
    lo,hi=gain['person_cluster_95_interval']
    text+=f'<p><strong>Random forest provides modest predictive value for unseen students.</strong> On 1,530 reports from 158 cohort-aware IDs, its MAE is {s["Random forest"]["mae"]:.2f}, versus {s["Population mean"]["mae"]:.2f} for the training-population mean: {100*gain["mae_improvement"]/s["Population mean"]["mae"]:.1f}% lower error. The paired improvement is {gain["mae_improvement"]:.2f} points, with a fixed-prediction group-bootstrap 95% interval of {lo:.2f}–{hi:.2f}. Ridge MAE is {s["Ridge"]["mae"]:.2f}, worse than baseline. The previous eligible subset shows the same direction. These are exploratory internal results, not external validation.</p>'
    for protocol,scores in r['scores'].items():
        text+='<h2>'+html.escape(protocol)+'</h2><table><tr><th>Model</th><th>Reports / IDs</th><th>MAE ↓</th><th>RMSE ↓</th><th>R²</th><th>Equal-student MAE ↓</th></tr>'
        for name,v in scores.items():text+=f'<tr><td>{name}</td><td>{v["n"]} / {v["ids"]}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td><td>{v["equal_student_mae"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>95% group-bootstrap interval</th></tr>'
        for name,v in r['comparisons'][protocol].items():
            lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{name}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Interpretation and limits</h2><p>The primary test includes all cleaned reports and does not require any previous OSE or stress. The secondary test keeps the original historical eligibility rule for comparability, without using OSE as a predictor. Neither table is directly interchangeable with the previous chronological 980-report test. Historical non-OSE test-student measurements are allowed; the model does not train on any held-out student outcomes. Training dates may overlap or follow test dates. This is a student-transfer experiment, not a prospective calendar-time test.</p>'
    text+='<p>All identical numeric ID suffixes across cohorts are assigned together because physical identity across cohorts is unverified. This may conservatively merge different people. Imputation, encoding, scaling, and facial PCA fit only training students. Missing modalities and uncertain demographic collection timing remain limitations. No hyperparameter search or test-driven model selection was performed.</p><h2>Analysis specification</h2>'
    text+=''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(str(v))+'</p>' for k,v in r['plan'].items())
    text+='<p><a href="results.json">Full results</a> · <a href="predictions.csv">Held-out predictions</a> · <a href="fold_membership.csv">Student split audit</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
