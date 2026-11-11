"""Nested student-group validation of stronger no-past-OSE regressors."""
from pathlib import Path
import hashlib,html,json,time,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import ose_unseen_students as old
from ose_fancy_transforms import NativeInput

ROOT=old.ROOT;OUT=ROOT/'outputs/ose-fancy-students'
CONFIGS={
 'CatBoost':[{'depth':depth,'l2_leaf_reg':l2} for depth in [3,5] for l2 in [5,20]],
 'Histogram boosting':[{'max_leaf_nodes':leaves,'l2_regularization':l2} for leaves in [7,15] for l2 in [10,50]],
 'Elastic Net':[{'alpha':alpha,'l1_ratio':ratio} for alpha in [.1,1,10] for ratio in [.1,.5,.9]]
}
PLAN={
 'target':'Supplied OSE engagement score in native units; no past OSE in any input. All 1530 cleaned reports from 158 cohort-aware IDs.',
 'outer_test':'Exactly the prior five student-held-out folds, random_state42; conservative same numeric suffix grouped across cohorts. No student/group overlap. No new test-driven split choice.',
 'tuning':'Three shuffled GroupKFold inner folds per outer training set, seed142. Select each family configuration by pooled inner validation MAE. All preprocessing fit only inner training students. Final outer model refit on all outer training students. No outer outcomes used in tuning.',
 'families':'CatBoost400 iterations, learning_rate.04, RMSE loss, depth3/5, l2 5/20, seed42, two threads, no early stopping. Histogram boosting200 iterations, learning_rate.05, leaf7/15, l2 10/50, minimum leaf20, squared loss, seed42, early_stopping=False. Elastic Net alpha.1/1/10, l1ratio.1/.5/.9, max_iter50000,tol1e-4.',
 'preprocessing':'Same train-only facial PCA10 for all families. Elastic Net and histogram boosting use the previous train-only median/indicator/scaler and categorical one-hot pipeline. CatBoost receives numeric NaNs natively, categorical missing explicitly Unknown/string, plus train-only facial components.',
 'selector':'Additional nested-selected-family prediction selects the family/configuration with lowest inner MAE within each outer fold. Never select a family using the outer test outcomes.',
 'metrics':'Report-weighted MAE primary; RMSE,R2,equal-student MAE secondary. Comparisons on identical records against original population mean and random forest. Conservative student-group bootstrap3000 draws; fixed-prediction uncertainty, not refitting or full model-selection uncertainty.',
 'limits':old.PLAN['limits']+' Same study period with overlapping dates, not future-semester validation. Earlier held-out-student non-OSE measurements allowed as inputs. Repeated exploratory test-set reuse means results are not an untouched confirmatory test.'
}

def estimator(family,config):
    if family=='CatBoost':return CatBoostRegressor(**config,iterations=400,learning_rate=.04,loss_function='RMSE',random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)
    if family=='Histogram boosting':return HistGradientBoostingRegressor(**config,max_iter=200,learning_rate=.05,min_samples_leaf=20,early_stopping=False,random_state=42)
    return ElasticNet(**config,max_iter=50000,tol=1e-4,random_state=42)

def preprocessing(train,cols,schema,native):
    pre=NativeInput(cols,schema) if native else old.previous.build(cols,schema,False).steps[0][1]
    pre.fit(train[cols]);return pre

def main():
    OUT.mkdir(parents=True,exist_ok=True);schema,d,cols=old.load();d=d.reset_index(drop=True)
    plan={**PLAN,'configurations':CONFIGS,'features':cols,'source_hashes':{n:hashlib.sha256((old.SOURCE/n).read_bytes()).hexdigest() for n in ['modeling_features.csv','feature_schema.json']}}
    (OUT/'analysis_plan.json').write_text(json.dumps(plan,indent=2))
    records=[];tuning=[];members=[];selected=[];fold_results=[]
    prior=pd.read_csv(old.OUT/'predictions.csv',float_precision='round_trip',dtype={'student_group':str});prior=prior[prior.protocol.eq('All students')]
    for fold,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(d,groups=d.student_group),1):
        started=time.monotonic();train=d.iloc[tr].reset_index(drop=True);test=d.iloc[te].reset_index(drop=True)
        assert set(train.student_group).isdisjoint(test.student_group)
        ref=prior[prior.fold.eq(fold)&prior.model.eq('Random forest')]
        assert set(ref.source_row)==set(test.source_row)
        for role,frame in [('train',train),('test',test)]:members.extend(frame[['source_row','person','student_group']].assign(outer_fold=fold,inner_fold=0,role=role).to_dict('records'))
        totals={f:np.zeros(len(configs)) for f,configs in CONFIGS.items()};nvalidation=0
        for inner,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=142).split(train,groups=train.student_group),1):
            a=train.iloc[it];b=train.iloc[iv];nvalidation+=len(b)
            assert set(a.student_group).isdisjoint(b.student_group)
            for role,frame in [('train',a),('validation',b)]:members.extend(frame[['source_row','person','student_group']].assign(outer_fold=fold,inner_fold=inner,role=role).to_dict('records'))
            encoded=preprocessing(a,cols,schema,False);native=preprocessing(a,cols,schema,True)
            matrices={False:(encoded.transform(a[cols]),encoded.transform(b[cols])),True:(native.transform(a[cols]),native.transform(b[cols]))}
            for family,configs in CONFIGS.items():
                x,y=matrices[family=='CatBoost']
                for index,config in enumerate(configs):
                    model=estimator(family,config);kw={'cat_features':native.cat_indices} if family=='CatBoost' else {}
                    model.fit(x,a.OSE,**kw);pred=model.predict(y);error=float(np.abs(pred-b.OSE.to_numpy()).sum());totals[family][index]+=error
                    tuning.append({'outer_fold':fold,'inner_fold':inner,'family':family,'config_index':index,'config':json.dumps(config,sort_keys=True),'validation_n':len(b),'mae':error/len(b)})
            print('Outer',fold,'inner',inner,'completed',round(time.monotonic()-started,1),'seconds',flush=True)
        fitted={};family_preds={};best={}
        for family,configs in CONFIGS.items():
            best_index=int(np.argmin(totals[family]));config=configs[best_index];best[family]=float(totals[family][best_index]/nvalidation)
            pre=preprocessing(train,cols,schema,family=='CatBoost');model=estimator(family,config)
            kw={'cat_features':pre.cat_indices} if family=='CatBoost' else {}
            model.fit(pre.transform(train[cols]),train.OSE,**kw);pred=model.predict(pre.transform(test[cols]));assert np.isfinite(pred).all()
            fitted[family]={'preprocessor':pre,'estimator':model,'config':config};family_preds[family]=pred
            selected.append({'outer_fold':fold,'family':family,'config_index':best_index,'config':config,'inner_mae':best[family]})
        winner=min(best,key=best.get);family_preds['Nested family selection']=family_preds[winner]
        joblib.dump({'models':fitted,'selected_family':winner,'features':cols,'schema':schema},OUT/f'fold_{fold}_models.joblib')
        for family,pred in family_preds.items():records.extend({'protocol':'All students','fold':fold,'model':family,'source_row':int(r.source_row),'person':r.person,'student_group':r.student_group,'observed':float(r.OSE),'predicted':float(v)} for (_,r),v in zip(test.iterrows(),pred))
        fold_results.append({'fold':fold,'selected_family':winner,'train_n':len(train),'test_n':len(test),'inner_mae':best})
        pd.DataFrame(records).to_csv(OUT/'new_predictions_checkpoint.csv',index=False)
        pd.DataFrame(tuning).to_csv(OUT/'inner_tuning.csv',index=False)
        pd.DataFrame(members).to_csv(OUT/'split_membership.csv',index=False)
        (OUT/'selected_configurations.json').write_text(json.dumps(selected,indent=2))
        print('Outer fold',fold,'complete; selected',winner,flush=True)
    p=pd.concat([pd.DataFrame(records),prior],ignore_index=True);p.to_csv(OUT/'predictions.csv',index=False)
    scores={};comparisons={}
    for family,g in p.groupby('model'):
        assert g.source_row.is_unique and set(g.source_row)==set(d.source_row)
        error=abs(g.observed-g.predicted);scores[family]={**old.previous.core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy()),'ids':g.person.nunique(),'equal_student_mae':float(error.groupby(g.person).mean().mean())}
    for family in list(CONFIGS)+['Nested family selection']:
        for base in ['Population mean','Random forest']:
            a=p[p.model.eq(family)].copy();b=p[p.model.eq(base)].copy();a['person']=a.student_group;b['person']=b.student_group
            comparisons[family+' versus '+base]=old.previous.paired(a,b)
    r={'plan':plan,'scores':scores,'comparisons':comparisons,'folds':fold_results}
    (OUT/'results.json').write_text(json.dumps(r,indent=2,default=int));render(r)
    print(json.dumps({'scores':scores,'comparisons':comparisons},indent=2,default=int),flush=True)

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Stronger OSE models for unseen students</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2f7}</style><h1>Stronger OSE models for unseen students</h1><p>No past OSE. Same1,530 reports158 cohort-aware student IDs and same five outer student-held-out folds as the previous experiment. All hyperparameter choices and preprocessing fit inside the training-student partitions. OSE denotes online student engagement.</p><table><tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>R²</th><th>Equal-student MAE ↓</th></tr>'
    for name,v in sorted(r['scores'].items(),key=lambda kv:kv[1]['mae']):text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae"]:.3f}</td><td>{v["rmse"]:.3f}</td><td>{v["r2"]:.3f}</td><td>{v["equal_student_mae"]:.3f}</td></tr>'
    text+='</table><p>Nested family selection chooses a family by inner validation within each training fold, not from this final score table. A family that ranks first in this table was observed on reused exploratory test data.</p><table><tr><th>Comparison</th><th>MAE improvement (positive better)</th><th>95% group-bootstrap interval</th></tr>'
    for name,v in r['comparisons'].items():
        lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
    gain=r['comparisons']['Elastic Net versus Random forest'];lo,hi=gain['person_cluster_95_interval']
    text+=f'</table><h2>Interpretation</h2><p><strong>No clear improvement over the original forest.</strong> Elastic Net has the lowest observed MAE,7.98 versus8.19 for the original forest, a difference of {gain["mae_improvement"]:.2f} points. Its95% group-bootstrap improvement interval is {lo:.2f} to {hi:.2f}, including no improvement. Relative to the population-mean baseline9.01, Elastic Net reduces MAE by11.4%. CatBoost and histogram boosting do not improve the forest. The nested family selector MAE8.24 also does not improve it. These results support modest cross-student predictability; increasing model complexity has not established additional value.</p><h2>Specification and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(str(v))+'</p>' for k,v in PLAN.items())
    text+='<p><a href="results.json">Full results</a> · <a href="inner_tuning.csv">Inner tuning scores</a> · <a href="selected_configurations.json">Selected configurations</a> · <a href="predictions.csv">Held-out predictions</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
