"""Student-held-out OSE prediction with earlier scores allowed at inference."""
import hashlib,html,json
import joblib
import numpy as np
import pandas as pd
import ose_unseen_students as old

OUT=old.ROOT/'outputs/ose-unseen-with-history'
PLAN={
 'question':'Can models trained on other students predict a held-out student later OSE when that student earlier OSE is supplied as an input?',
 'distinction':'No held-out student contributes any training outcome. Earlier held-out OSE can be an input for a later response from the same held-out student. This is new-to-training, with observed personal history; not a student with no earlier OSE.',
 'primary':'Original eligible subset1118 reports138 cohort-aware IDs, with at least2 earlier OSE and1 earlier PSS. Exact same five group folds as the no-OSE student-held-out experiment.',
 'secondary':'All1530 cleaned reports, including missing OSE history; missing numerical fields training-median imputed with indicators. Last/past-mean OSE baselines fall back to fold training population mean when unavailable.',
 'models':'Fixed prior ridge alpha10 and random forest300trees depth3 minimumleaf10 seed42. Test past-OSE-only and all-predictors including past OSE. No tuning.',
 'history':'Seven previously verified OSE summaries, only same-person reports strictly before target date minus7days. Current response and future responses never inputs. Earlier non-OSE features retain original availability buffers.',
 'splits':'Reuse saved prior student memberships exactly. Conservative identical numeric suffixes grouped across cohorts. Fit every preprocessing stage on training students only.',
 'metrics':'Same-row report-weighted MAE,RMSE,R2 plus equal-student MAE. Compare history-added against same fixed algorithm without OSE, history-only, and last-score baseline. Fixed-prediction conservative-student-group bootstrap3000draws.',
 'limits':old.PLAN['limits']+' Training/test dates overlap; not unseen-semester forecasting. Own earlier scores allowed at inference, so personal-history availability is required for primary result. Repeated exploratory dataset reuse.'
}

def main():
    OUT.mkdir(parents=True,exist_ok=True);schema,d,other=old.load();history=old.previous.core.OSE_FEATURES
    cols=history+other;indexed=d.set_index('source_row',drop=False)
    members=pd.read_csv(old.OUT/'fold_membership.csv',dtype={'student_group':str})
    members.to_csv(OUT/'fold_membership.csv',index=False)
    prior=pd.read_csv(old.OUT/'predictions.csv',float_precision='round_trip',dtype={'student_group':str})
    plan={**PLAN,'features':{'Past OSE only':history,'All including past OSE':cols},'source_hashes':{n:hashlib.sha256((old.SOURCE/n).read_bytes()).hexdigest() for n in ['modeling_features.csv','feature_schema.json']}}
    (OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    records=[]
    for (protocol,fold),mm in members.groupby(['protocol','fold']):
        train=indexed.loc[mm.loc[mm.role.eq('train'),'source_row']];test=indexed.loc[mm.loc[mm.role.eq('test'),'source_row']]
        assert set(train.student_group).isdisjoint(test.student_group)
        preds={'Last OSE':test.ose_last.fillna(train.OSE.mean()).to_numpy(),'Past mean OSE':test.ose_mean.fillna(train.OSE.mean()).to_numpy()};models={}
        for scope,columns in [('Past OSE only',history),('All including past OSE',cols)]:
            for forest,family in [(False,'Ridge'),(True,'Random forest')]:
                name=scope+' | '+family;model=old.previous.build(columns,schema,forest).fit(train[columns],train.OSE)
                models[name]={'features':columns,'model':model};preds[name]=model.predict(test[columns])
        joblib.dump(models,OUT/(('all' if protocol=='All students' else 'eligible')+f'_fold_{fold}.joblib'))
        for name,values in preds.items():
            assert np.isfinite(values).all()
            records.extend({'protocol':protocol,'fold':int(fold),'model':name,'source_row':int(r.source_row),'person':r.person,'student_group':r.student_group,'observed':float(r.OSE),'predicted':float(v),'past_OSE_reports':int(r.ose_count)} for (_,r),v in zip(test.iterrows(),values))
        print(protocol,'fold',fold,'done',flush=True)
    prior['model']=prior.model.replace({'Ridge':'Without past OSE | Ridge','Random forest':'Without past OSE | Random forest'})
    p=pd.concat([pd.DataFrame(records),prior],ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False)
    scores={};comparisons={}
    for protocol,pp in p.groupby('protocol'):
        scores[protocol]={};comparisons[protocol]={}
        for name,g in pp.groupby('model'):
            error=abs(g.predicted-g.observed)
            scores[protocol][name]={**old.previous.core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':g.person.nunique(),'equal_student_mae':float(error.groupby(g.person).mean().mean())}
        for family in ['Ridge','Random forest']:
            name='All including past OSE | '+family
            for baseline in ['Without past OSE | '+family,'Past OSE only | '+family,'Last OSE']:
                a=pp[pp.model.eq(name)].copy();b=pp[pp.model.eq(baseline)].copy();a.person=a.student_group;b.person=b.student_group
                comparisons[protocol][name+' versus '+baseline]=old.previous.paired(a,b)
    result={'plan':plan,'scores':scores,'comparisons':comparisons}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=int));render(result)
    print(json.dumps(scores,indent=2,default=int),flush=True)

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Unseen students with OSE history</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2f7}</style><h1>Unseen students, with earlier OSE as an input</h1><p>Every held-out student is entirely excluded from model training. At prediction time, their own earlier scores can be supplied as inputs. No current/future score is used. OSE denotes online student engagement.</p>'
    primary=r['scores']['Previous eligible subset'];gain=r['comparisons']['Previous eligible subset']['All including past OSE | Random forest versus Without past OSE | Random forest'];lo,hi=gain['person_cluster_95_interval']
    text+=f'<p><strong>Earlier OSE remains useful for students unseen during training.</strong> On the same1,118 history-eligible reports138 held-out student IDs, adding personal OSE history reduces all-variable forest MAE from {primary["Without past OSE | Random forest"]["mae"]:.2f} to {primary["All including past OSE | Random forest"]["mae"]:.2f}. Paired gain {gain["mae_improvement"]:.2f},95% group-bootstrap interval {lo:.2f} to {hi:.2f}. A ridge model using only earlier OSE summaries has the lowest observed MAE {primary["Past OSE only | Ridge"]["mae"]:.2f}, versus {primary["Last OSE"]["mae"]:.2f} for simply using the last earlier score. Adding the other predictors does not improve on the corresponding history-only model. These results require personal history at prediction time.</p>'
    for protocol in ['Previous eligible subset','All students']:
        text+='<h2>'+html.escape(protocol)+'</h2><table><tr><th>Model</th><th>Reports / IDs</th><th>MAE ↓</th><th>RMSE ↓</th><th>R²</th></tr>'
        for name,v in sorted(r['scores'][protocol].items(),key=lambda kv:kv[1]['mae']):text+=f'<tr><td>{html.escape(name)}</td><td>{v["n"]} / {v["ids"]}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>95% group-bootstrap interval</th></tr>'
        for name,v in r['comparisons'][protocol].items():
            lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Method and limitations</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(str(v))+'</p>' for k,v in PLAN.items())
    text+='<p><a href="results.json">Full results</a> · <a href="predictions.csv">Held-out predictions</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
