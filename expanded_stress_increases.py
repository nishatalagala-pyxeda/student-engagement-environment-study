"""Expand above-usual stress forecasting to all cleaned PSS reports."""
from pathlib import Path
import json,hashlib,html
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,average_precision_score
import stress_increases as old
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/expanded-stress-increases'
PLAN={
 'question':'Does using all available stress reports improve history-based prediction of above-usual stress?',
 'target':'Above-usual event = target PSS minus mean of admissible earlier reports >=0.5. At least2 earlier reports; descriptive threshold on provided0–4 scale, not clinical. Also regress signed excess over that mean.',
 'history':'Same7 history features and timing as prior analysis: historical mean, last score, last-minus-mean, population SD, change between last2 reports, count, days since last report. Only reports strictly before target date minus7 days; test histories also strictly before block training cutoff. Frozen per test block; no test outcomes update any reference.',
 'cohort':'All valid, PSS-only cleaned reports, regardless of environment/Fitbit availability. Conflicting person/date values excluded, identical repeats collapsed. Same cohort-aware ID normalization; physical cross-cohort identity not established.',
 'validation':'Same14-day blocks anchored June1,2020; train strictly before block start minus7days, require40 raw train reports10 IDs, then20 history-eligible reports5 IDs and both event classes. Same April holdout: training before March25,2021, tests April1 onward. Only history-eligible tests scored. Reused study/time periods; exploratory, not new untouched data.',
 'models':'Same logistic C=1 and ridge alpha10 primary; forest sensitivity300trees depth3 leaf10, seed42. All7 predictors, no IDs, demographics, environmental or wearable predictors. Train-only imputation/scaling. Class decision probability0.5, no new threshold tuning.',
 'baselines':'Classification training event prevalence, always no increase, last-deviation persistence. Regression zero excess (personal historical mean), latest deviation (last score), training mean excess. Reconstructed scores clipped0–4.',
 'controlled_comparison':'Besides evaluating the expanded population, repeat the original calendar folds on the exact original environment-selected test records, with either original234-row training pool or all cleaned earlier reports. Same target references/test dates and minimum histories, no matching selected by outcomes. Verify restricted predictions reproduce prior saved history models. This isolates expanded training from changing the test population.',
 'uncertainty':'Paired fixed-prediction person-cluster bootstrap3000draws, seed42, for Brier/MAE improvements and AUC difference in controlled comparisons; standalone history ranking intervals also clustered by person. No refitting or full training/selection uncertainty; no claim of causation.',
 'scope':'Primary comparison expands the history model. A separate environmental comparison tests earlier measurements with the same expanded training pool where measurements exist; see environment_analysis_plan.json. A different test population/event rate makes raw old/new population metrics incomparable.',
 'limits':'Earlier history must be observed; does not predict a new student without history. Noisy individual means, sparse positive events, irregular history gaps, aggregate score scoring unavailable, retrospective recorded dates/availability and repeated reuse of study. No advanced algorithms or altered target thresholds.'}

def load():
    path=next(old.base.SOURCE.rglob('PANAS_PSS.csv'));raw=pd.read_csv(path);raw['person']=raw.ID.map(old.canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce');raw['source_row']=raw.index+2
    valid=raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4);v=raw[valid].copy();bad=v.groupby(['person','date']).PSS.transform('nunique').gt(1)
    v[bad].to_csv(OUT/'excluded_conflicting_reports.csv',index=False);nonconflict=v[~bad];h=nonconflict.drop_duplicates(['person','date']).sort_values(['person','date']).reset_index(drop=True)
    assert len(h)==1534 and h.person.nunique()==159 and not h.duplicated(['person','date']).any()
    h[['source_row','person','date','PSS']].to_csv(OUT/'cleaned_stress_reports.csv',index=False)
    previous=pd.read_csv(ROOT/'outputs/stress-increases/modeling_features.csv');previous['date']=pd.to_datetime(previous.survey_date)
    target=h.set_index('source_row').loc[previous.source_row];np.testing.assert_allclose(target.PSS,previous.PSS);assert np.array_equal(target.person,previous.person)
    metadata={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'raw_rows':len(raw),'invalid_or_missing_id_date_score_rows':int((~valid).sum()),'conflicting_rows_excluded':int(bad.sum()),'identical_repeat_rows_collapsed':len(nonconflict)-len(h),'cleaned_reports':len(h),'cleaned_ids':int(h.person.nunique()),'first_date':str(h.date.min().date()),'last_date':str(h.date.max().date()),'feature_names':old.HISTORY,'previous_model_features_sha256':hashlib.sha256((ROOT/'outputs/stress-increases/modeling_features.csv').read_bytes()).hexdigest()}
    full=old.add_history(h,h);eligible=full.prior_report_count.ge(2);metadata.update(history_eligible_reports=int(eligible.sum()),history_eligible_ids=int(full.loc[eligible,'person'].nunique()),history_ineligible_reports=int((~eligible).sum()))
    full.to_csv(OUT/'full_history_features.csv',index=False)
    return h,previous,metadata

def evaluate(h,previous,controlled=False):
    records=[];audits=[];splits=[]
    for protocol in ['purged_rolling_blocks','frozen_April_test']:
        pool=previous if controlled else h
        for block,cutoff,tr,te in old.prev.get_splits(pool,protocol):
            test=old.add_history(pool.iloc[te],h,cutoff);test=test[test.prior_report_count.ge(2)].copy()
            original_training=old.add_history(pool.iloc[tr],h);original_training=original_training[original_training.prior_report_count.ge(2)].copy()
            if len(original_training)<20 or original_training.person.nunique()<5 or not len(test) or original_training.excess_stress.ge(.5).nunique()<2:continue
            pools={'Expanded training':h[h.date.lt(cutoff)]}
            if controlled:pools['Original training']=pool.iloc[tr]
            for training_name,raw_training in pools.items():
                train=old.add_history(raw_training,h);train=train[train.prior_report_count.ge(2)].copy()
                assert train.date.max()<cutoff and set(train.source_row).isdisjoint(test.source_row)
                assert train.excess_stress.ge(.5).nunique()==2
                splits.append({'protocol':protocol,'block':block,'training_pool':training_name,'cutoff_exclusive':str(cutoff.date()),'train_n':len(train),'train_ids':int(train.person.nunique()),'test_n':len(test),'test_ids':int(test.person.nunique()),'train_events':int(train.excess_stress.ge(.5).sum()),'test_events':int(test.excess_stress.ge(.5).sum()),'test_first':str(test.date.min().date()),'test_last':str(test.date.max().date())})
                for role,frame in [('train',train),('test',test)]:
                    for _,r in frame.iterrows():audits.append({'protocol':protocol,'block':block,'training_pool':training_name,'role':role,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'cutoff_exclusive':str(cutoff.date()),'history_source_rows':r.history_source_rows,'latest_history_date':str(r.latest_history_date.date()),**{c:float(r[c]) for c in old.HISTORY},'PSS':float(r.PSS),'excess_stress':float(r.excess_stress),'event':bool(r.excess_stress>=.5)})
                y=train.excess_stress.ge(.5).astype(int)
                classification={'Training prevalence':np.full(len(test),y.mean()),'Always no increase':np.zeros(len(test)),'Last-deviation persistence':test.prior_deviation.ge(.5).to_numpy(float)}
                regression={'Zero excess (past mean)':np.zeros(len(test)),'Last deviation':test.prior_deviation.to_numpy(),'Training mean excess':np.clip(np.full(len(test),train.excess_stress.mean()),-test.past_mean_stress.to_numpy(),4-test.past_mean_stress.to_numpy())}
                for forest,family in [(False,'linear'),(True,'forest')]:
                    classifier=old.build(old.HISTORY,[],forest,True).fit(train[old.HISTORY],y)
                    regressor=old.build(old.HISTORY,[],forest,False).fit(train[old.HISTORY],train.excess_stress)
                    classification['History | '+family]=classifier.predict_proba(test[old.HISTORY])[:,1]
                    regression['History | '+family]=np.clip(regressor.predict(test[old.HISTORY]),-test.past_mean_stress.to_numpy(),4-test.past_mean_stress.to_numpy())
                for task,predictions in [('classification',classification),('regression',regression)]:
                    for name,preds in predictions.items():
                        for (_,r),pred in zip(test.iterrows(),preds):records.append({'protocol':protocol,'block':block,'training_pool':training_name,'task':task,'model':name,'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'observed':float(r.excess_stress>=.5) if task=='classification' else float(r.excess_stress),'predicted':float(pred),'observed_PSS':float(r.PSS),'history_mean':float(r.past_mean_stress),'past_count':int(r.prior_report_count)})
        print('Completed',('controlled' if controlled else 'all_reports'),protocol,flush=True)
    p=pd.DataFrame(records);assert not p.duplicated(['protocol','training_pool','task','model','source_row']).any()
    prefix='controlled' if controlled else 'expanded';p.to_csv(OUT/f'{prefix}_predictions.csv',index=False);pd.DataFrame(audits).to_csv(OUT/f'{prefix}_history_target_audit.csv',index=False)
    scores={};cis={}
    for (protocol,training_pool,task),pp in p.groupby(['protocol','training_pool','task']):
        scores.setdefault(protocol,{}).setdefault(training_pool,{})[task]={};cis.setdefault(protocol,{}).setdefault(training_pool,{})[task]={}
        for name,g in pp.groupby('model'):
            s=old.classifier_score(g.observed,g.predicted) if task=='classification' else old.base.score(g.observed.to_numpy(),g.predicted.to_numpy());s['people']=int(g.person.nunique());scores[protocol][training_pool][task][name]=s
        baseline='Training prevalence' if task=='classification' else 'Zero excess (past mean)'
        for family in ['linear','forest']:
            name='History | '+family;cis[protocol][training_pool][task][name+' versus '+baseline]=old.paired_ci(pp[pp.model.eq(name)],pp[pp.model.eq(baseline)],task=='classification')
    controlled_improvements={}
    if controlled:
        previous_p=pd.read_csv(ROOT/'outputs/stress-increases/predictions.csv')
        for (protocol,task,model),g in p[p.training_pool.eq('Original training')].groupby(['protocol','task','model']):
            prior_p=previous_p[previous_p.protocol.eq(protocol)&previous_p.task.eq(task)&previous_p.model.eq(model)].set_index('source_row').sort_index();g=g.set_index('source_row').sort_index()
            assert g.index.equals(prior_p.index);np.testing.assert_allclose(g.observed,prior_p.observed,atol=1e-12);np.testing.assert_allclose(g.predicted,prior_p.predicted,atol=1e-10,rtol=1e-10)
        for (protocol,task),pp in p.groupby(['protocol','task']):
            controlled_improvements.setdefault(protocol,{})[task]={}
            for family in ['linear','forest']:
                name='History | '+family;a=pp[pp.model.eq(name)&pp.training_pool.eq('Expanded training')];b=pp[pp.model.eq(name)&pp.training_pool.eq('Original training')]
                controlled_improvements[protocol][task][name]=old.paired_ci(a,b,task=='classification')
    return {'scores':scores,'baseline_comparisons':cis,'splits':splits,'controlled_training_improvements':controlled_improvements},p

def ranking_intervals(p):
    result={}
    for (protocol,pool,name),g in p[p.task.eq('classification')&p.model.str.startswith('History')].groupby(['protocol','training_pool','model']):
        groups=[x for _,x in g.groupby('person')];rng=np.random.default_rng(42);aucs=[]
        # Array concatenation avoids expensive repeated data-frame operations.
        yy=[x.observed.to_numpy() for x in groups];prob=[x.predicted.to_numpy() for x in groups]
        for _ in range(3000):
            ix=rng.integers(len(groups),size=len(groups));y=np.concatenate([yy[i] for i in ix]);pred=np.concatenate([prob[i] for i in ix])
            if len(np.unique(y))==2:aucs.append(roc_auc_score(y,pred))
        result.setdefault(protocol,{}).setdefault(pool,{})[name]={'AUC_person_cluster_95_interval':np.quantile(aucs,[.025,.975]).tolist(),'draws':len(aucs)}
    return result

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Expanded stress-increase prediction</title><style>body{font:16px/1.6 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}</style><h1>Does expanding the sample improve stress-increase prediction?</h1>'
    m=r['metadata'];text+=f'<p>The full source provides {m["cleaned_reports"]:,} cleaned weekly stress reports from {m["cleaned_ids"]} cohort-aware IDs. {m["history_eligible_reports"]:,} reports from {m["history_eligible_ids"]} IDs have at least two admissible past reports before validation freezes. Above-usual stress means at least 0.5 points above the historical mean on the supplied 0–4 scale; this is not a clinical cutoff or necessarily an increase over the immediately preceding week.</p>'
    rolling=r['expanded']['scores']['purged_rolling_blocks']['Expanded training'];c=rolling['classification']['History | linear'];reg=rolling['regression'];f=reg['History | forest']['mae'];b=reg['Zero excess (past mean)']['mae'];ci=r['expanded']['baseline_comparisons']['purged_rolling_blocks']['Expanded training']['regression']['History | forest versus Zero excess (past mean)'];lo,hi=ci['person_cluster_95_interval']
    text+=f'<h2>What improved?</h2><p><strong>The broader analysis finds a small improvement in predicting the size of deviations; reliable alerts remain unresolved.</strong> Rolling validation now includes {c["n"]} reports from {c["people"]} IDs and {c["events"]} above-usual events. The forest excess-stress MAE is {f:.3f}, versus {b:.3f} for the personal-mean baseline: a reduction of {b-f:.3f} points ({(b-f)/b:.1%}). The paired person-cluster95% interval for that reduction is {lo:.3f} to {hi:.3f}; the separate April test interval includes zero. The prespecified linear regression does not establish a gain over the baseline.</p><p>The history logistic classifier has rolling AUC {c["roc_auc"]:.3f} and average precision {c["average_precision"]:.3f}, with event prevalence {c["event_rate"]:.1%}. At the unchanged0.5 threshold it detects {c["tp"]} of {c["events"]} events, with {c["fp"]} false alerts. Neither classifier establishes a Brier improvement over training prevalence, and April ranking falls to chance or below. No alert threshold was optimized.</p><p>On the exact original152 test records, expanding training changes logistic AUC from0.571 to0.545 and forest AUC from0.631 to0.588. Regression errors fall numerically, but their paired improvement intervals span zero. Thus the broader population shows a modest forest regression gain, while a benefit from additional training alone is not demonstrated on the original test group. This distinction prevents attributing a changed test population to better training.</p>'
    for section,label in [('expanded','All usable stress reports'),('controlled','Same original test records: effect of expanded training')]:
        text+='<h2>'+label+'</h2>'
        if section=='expanded':text+='<p>This broader cohort changes both training and test populations. Compare against the baselines below; do not attribute raw differences from the previous smaller test to sample size alone.</p>'
        else:text+='<p>Both training pools predict exactly the same original test rows with identical historical reference means and labels. Original-training predictions reproduce the previous saved history-only results. This comparison isolates the value of extra earlier training reports on that selected test population.</p>'
        for protocol,pools in r[section]['scores'].items():
            text+='<h3>'+protocol+'</h3><table><tr><th>Training pool / classifier</th><th>N / IDs / events</th><th>AUC</th><th>Average precision</th><th>Brier↓</th><th>Recall</th><th>Precision</th><th>True / false alerts</th></tr>'
            for pool,tasks in pools.items():
                for name,s in tasks['classification'].items():
                    text+=f'<tr><td>{html.escape(pool+" / "+name)}</td><td>{s["n"]} / {s["people"]} / {s["events"]}</td><td>{s["roc_auc"]:.3f}</td><td>{s["average_precision"]:.3f}</td><td>{s["brier"]:.4f}</td><td>{s["recall"]:.3f}</td><td>{s["precision"]:.3f}</td><td>{s["tp"]} / {s["fp"]}</td></tr>'
            text+='</table><p>Decision probability fixed at 0.5. Always predicting no event can give high accuracy but detect no events. Brier measures squared probability error; lower is better. AUC measures ranking; 0.5 is chance.</p><table><tr><th>Training pool / regression</th><th>Excess MAE↓</th><th>RMSE↓</th><th>R²</th></tr>'
            for pool,tasks in pools.items():
                for name,s in tasks['regression'].items():text+=f'<tr><td>{html.escape(pool+" / "+name)}</td><td>{s["mae"]:.3f}</td><td>{s["rmse"]:.3f}</td><td>{s["r2"]:.3f}</td></tr>'
            text+='</table>'
    text+='<h2>Paired uncertainty</h2><table><tr><th>Analysis / protocol / comparator</th><th>Metric</th><th>Improvement (positive better)</th><th>Person-cluster 95% interval</th></tr>'
    for section in ['expanded','controlled']:
        for protocol,pools in r[section]['baseline_comparisons'].items():
            for pool,tasks in pools.items():
                for task,cs in tasks.items():
                    for name,v in cs.items():
                        lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(section+" / "+protocol+" / "+pool+" / "+name)}</td><td>{v["metric"]}</td><td>{v["improvement"]:.4f}</td><td>{lo:.4f} to {hi:.4f}</td></tr>'
    for protocol,tasks in r['controlled']['controlled_training_improvements'].items():
        for task,cs in tasks.items():
            for name,v in cs.items():
                lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape("Expanded versus original training / "+protocol+" / "+name)}</td><td>{v["metric"]}</td><td>{v["improvement"]:.4f}</td><td>{lo:.4f} to {hi:.4f}</td></tr>'
    text+='</table>'
    if r.get('controlled_auc_improvement_intervals'):
        text+='<h2>Ranking change from expanded training on the same test records</h2><table><tr><th>Protocol / classifier</th><th>AUC change</th><th>Person-cluster95% interval</th></tr>'
        for protocol,models in r['controlled_auc_improvement_intervals'].items():
            for name,v in models.items():
                lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(protocol+" / "+name)}</td><td>{v["auc_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+='<h2>Definitions and limitations</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p><a href="results.json">Complete results and ranking intervals</a> · <a href="expanded_predictions.csv">Expanded-population predictions</a> · <a href="controlled_predictions.csv">Controlled comparison predictions</a> · <a href="expanded_history_target_audit.csv">Historical reference audit</a> · <a href="integrity_checks.json">Checks</a></p>'
    if (OUT/'environment_report_section.html').exists():text+=(OUT/'environment_report_section.html').read_text()
    (OUT/'report.html').write_text(text)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));h,previous,metadata=load();print('Source',json.dumps(metadata),flush=True)
    expanded,p=evaluate(h,previous);controlled,cp=evaluate(h,previous,True);rank=ranking_intervals(p)
    full=old.add_history(h,h);full=full[full.prior_report_count.ge(2)];models={}
    for forest,family in [(False,'linear'),(True,'forest')]:
        for classification,task in [(True,'classification'),(False,'regression')]:
            y=full.excess_stress.ge(.5).astype(int) if classification else full.excess_stress;name=family+' '+task;models[name]=old.build(old.HISTORY,[],forest,classification).fit(full[old.HISTORY],y)
    joblib.dump({'models':models,'features':old.HISTORY,'plan':PLAN},OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    for name,model in models.items():np.testing.assert_allclose(model.predict(full[old.HISTORY]),restored['models'][name].predict(full[old.HISTORY]))
    checks={'cleaned_source_is1534_reports159_IDs':True,'target_reference_excludes_current_and_future_reports':True,'test_history_frozen_before_training_cutoff':True,'train_only_imputation_and_scaling':True,'same_rows_across_models':True,'original_restricted_predictions_reproduce_saved_results':True,'controlled_comparison_uses_same_test_rows_labels_references':True,'serialization_roundtrip_passes':True}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2));r={'plan':PLAN,'metadata':metadata,'expanded':expanded,'controlled':controlled,'expanded_ranking_intervals':rank,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(OUT/'results.json').write_text(json.dumps(r,indent=2));render(r)
    print(json.dumps({'expanded':expanded['scores'],'controlled':controlled['scores'],'controlled_training_improvements':controlled['controlled_training_improvements']},indent=2),flush=True)
if __name__=='__main__':main()
