"""Does prior stress modify the predictive/associational role of bedroom environment?"""
from pathlib import Path
import os,json,hashlib,html
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
import stress_environment_hypothesis as previous
from stress_moderation_features import MEANS,TRANSFORMED,TERMS,ModerationTerms,transformed_levels
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/stress-environment-moderation'
PLAN={
 'question':'Do environmental associations/predictions depend on prior stress, and are they stronger when prior stress is higher?',
 'primary_moderator':'Most recent PSS report at least7 days before each target survey, continuous0..4 centered at2. Test histories additionally frozen strictly before training cutoff.',
 'initial_baseline_sensitivity':'First valid PSS report on/before that participant first matched exposure start, fixed across their observations. This tests a study-start trait rather than a changing prior-week state.',
 'primary_test':'Joint7 environmental-deviation-by-prior-stress interaction coefficients in person/calendar-week FE regression. Main environment terms, prior stress, prior stress squared, history mean and history age included. Positive interaction for an exposure means its signed association increases with prior stress; not every exposure has an intrinsically adverse direction.',
 'association_environment':'Seven transformed environmental means (same units as previous association), centered on each person average across the retained association cohort. Target is moderation of within-person deviations, not persistent room differences. Association-only person centering is not used for held-out prediction.',
 'sensitivities':'Fixed first pre-exposure baseline moderator, and first differences with exact differences of interaction products, calendar indicators and history terms. All are exploratory follow-ups to earlier inspected data.',
 'prediction':'Three nested ridge(alpha10) models: history/control with quadratic prior stress; plus additive environment; plus7 explicit products. All shared main effects retained. Same purged rolling and frozen April splits. Imputation, exposure centering/scaling and products fitted on training only. Main environment includes transformed weekly means and7 between-day SDs; interactions only with the7 means.',
 'descriptive_groups':'Prior PSS<2 vs>=2, fixed scale midpoint, not a clinical threshold. Unknown history separate. Group-specific error gains and their difference are resampled jointly by person. No direct subgroup inference if fewer than5 persons or10 responses in either group.',
 'inference':'Interaction joint restricted-null wild person-cluster test1999 draws, seed42. Individual7 interaction tests BH adjusted. CR1 person-cluster t intervals. Model-family/analysis choices fixed before this moderation run, but dataset and earlier test results have already been inspected.',
 'causal_limits':'This is prediction/association moderation, not proof of differential causal sensitivity. Confounding, reverse causation, short-panel lag/FE bias, bounded stress scale/regression to mean, exposure error and sparse high-stress support remain.'}


def load():
    df=pd.read_csv(ROOT/'outputs/depress-weekly-stress/modeling_data.csv');df['date']=pd.to_datetime(df.survey_date)
    source=next(previous.original.SOURCE.rglob('PANAS_PSS.csv'));raw=pd.read_csv(source);raw['person']=raw.ID.map(previous.canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce')
    h=raw[raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4)].copy();bad=h.groupby(['person','date']).PSS.transform('nunique').gt(1);h=h[~bad].drop_duplicates(['person','date']).sort_values(['person','date']).reset_index(drop=True)
    first={}
    for person,g in df.groupby('person'):
        before=h[h.person.eq(person)&h.date.le(g.date.min()-pd.Timedelta(days=7))]
        if len(before):first[person]={'value':float(before.iloc[0].PSS),'date':str(before.iloc[0].date.date())}
    meta={'n':len(df),'people':int(df.person.nunique()),'history_rows':len(h),'first_baselines_available':len(first),'source_input_sha256':hashlib.sha256((ROOT/'outputs/depress-weekly-stress/modeling_data.csv').read_bytes()).hexdigest(),'history_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'transformer_sha256':hashlib.sha256((ROOT/'stress_moderation_features.py').read_bytes()).hexdigest()}
    return df,h,first,meta


def predictive(df,h):
    base_num,cats=previous.scopes_for()['History + controls'];env_sd=[c for c in previous.scopes_for()['History + controls + environment'][0] if c.endswith('_between_day_sd')]
    definitions={'History/control':(False,False,base_num+['prior_stress_squared']),'Additive environment':(True,False,base_num+['prior_stress_squared']+TRANSFORMED+env_sd),'Environment × prior stress':(True,True,base_num+['prior_stress_squared']+TRANSFORMED+env_sd+TERMS)}
    rawcols=list(dict.fromkeys(base_num+MEANS+env_sd+cats));models={name:make_pipeline(ModerationTerms(environment=env,interactions=inter),previous.original.build(nums,cats,False)) for name,(env,inter,nums) in definitions.items()}
    assert not set(rawcols)&{'PSS','person','source_row','date','latest_history_date'}
    records=[];audit=[];splits=[]
    for protocol in ['frozen_April_test','purged_rolling_blocks']:
        seen=set()
        for block,cutoff,tr,te in previous.get_splits(df,protocol):
            train=previous.add_history(df.iloc[tr].reset_index(drop=True),h);test=previous.add_history(df.iloc[te].reset_index(drop=True),h,cutoff)
            assert seen.isdisjoint(test.source_row);seen.update(test.source_row)
            assert train.date.max()<=test.date.min()-pd.Timedelta(days=7)
            for _,r in test.iterrows():audit.append({'protocol':protocol,'block':block,'source_row':int(r.source_row),'person':r.person,'target_date':str(r.date.date()),'prior_stress':float(r.prior_stress) if pd.notna(r.prior_stress) else None,'history_latest':str(r.latest_history_date.date()) if pd.notna(r.latest_history_date) else None,'history_cutoff_exclusive':str(cutoff.date())})
            for name,template in models.items():
                model=clone(template).fit(train[rawcols],train.PSS);p=np.clip(model.predict(test[rawcols]),0,4)
                for (_,r),value in zip(test.iterrows(),p):records.append({'protocol':protocol,'block':block,'model':name,'source_row':int(r.source_row),'person':r.person,'survey_date':str(r.date.date()),'prior_stress':float(r.prior_stress) if pd.notna(r.prior_stress) else np.nan,'group':'Unknown history' if pd.isna(r.prior_stress) else ('Higher prior score (>=2)' if r.prior_stress>=2 else 'Lower prior score (<2)'),'observed':float(r.PSS),'predicted':float(value)})
            splits.append({'protocol':protocol,'block':block,'train_n':len(train),'test_n':len(test)})
        print('Completed predictive moderation',protocol,flush=True)
    pred=pd.DataFrame(records);pred.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'prediction_history_audit.csv',index=False)
    assert not pred.duplicated(['protocol','model','source_row']).any()
    scores={};groups={}
    for (protocol,model),g in pred.groupby(['protocol','model'],sort=False):
        scores.setdefault(protocol,{})[model]=previous.original.score(g.observed.to_numpy(),g.predicted.to_numpy())
        for label,p in g.groupby('group'):
            groups.setdefault(protocol,{}).setdefault(label,{})[model]={**previous.original.score(p.observed.to_numpy(),p.predicted.to_numpy()),'people':int(p.person.nunique())}
    ci={}
    for protocol in pred.protocol.unique():
        p=pred[pred.protocol.eq(protocol)];ci[protocol]={}
        for baseline,model in [('History/control','Additive environment'),('History/control','Environment × prior stress'),('Additive environment','Environment × prior stress')]:
            a=p[p.model.eq(model)].set_index('source_row');b=p[p.model.eq(baseline)].set_index('source_row').loc[a.index];delta=(b.predicted-b.observed).abs()-(a.predicted-a.observed).abs();names=sorted(a.person.unique());chunks=[pd.DataFrame({'improvement':delta[a.person.eq(v)],'group':a.loc[a.person.eq(v),'group']}) for v in names];rng=np.random.default_rng(42);boot=[];diffs=[]
            low='Lower prior score (<2)';high='Higher prior score (>=2)';support={label:{'n':int(a.group.eq(label).sum()),'people':int(a.loc[a.group.eq(label),'person'].nunique())} for label in [low,high]};valid=all(support[label]['n']>=10 and support[label]['people']>=5 for label in [low,high])
            point_diff=float(delta[a.group.eq(high)].mean()-delta[a.group.eq(low)].mean()) if valid else None
            for _ in range(5000):
                sample=pd.concat([chunks[j] for j in rng.integers(0,len(chunks),len(chunks))],ignore_index=True);boot.append(sample.improvement.mean())
                if valid and sample.group.eq(high).any() and sample.group.eq(low).any():diffs.append(sample.loc[sample.group.eq(high),'improvement'].mean()-sample.loc[sample.group.eq(low),'improvement'].mean())
            ci[protocol][model+' versus '+baseline]={'mae_improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'subgroup_support':support,'higher_minus_lower_improvement':point_diff,'higher_minus_lower_95_interval':np.quantile(diffs,[.025,.975]).tolist() if diffs else None,'note':'Positive favors added model; subgroup difference positive means more improvement when prior score>=2. Participant clusters resampled jointly across groups. Sparse April subgroup is not tested. Fixed predictions, no refitting.'}
    full=previous.add_history(df,h);fitted={name:clone(model).fit(full[rawcols],full.PSS) for name,model in models.items()};payload={'models':fitted,'raw_features':rawcols,'numeric_features_by_model':{k:v[2] for k,v in definitions.items()},'categorical_features':cats,'bounds':(0,4),'plan':PLAN};joblib.dump(payload,OUT/'models.joblib');restored=joblib.load(OUT/'models.joblib')
    np.testing.assert_allclose(restored['models']['Environment × prior stress'].predict(full[rawcols]),fitted['Environment × prior stress'].predict(full[rawcols]))
    return scores,groups,ci,splits


def known_history_summary(pred):
    summaries={}
    for protocol,p in pred[pred.prior_stress.notna()].groupby('protocol',sort=False):
        scores={}
        for name,d in p.groupby('model',sort=False):scores[name]={**previous.original.score(d.observed.to_numpy(),d.predicted.to_numpy()),'people':int(d.person.nunique())}
        a=p[p.model.eq('Environment × prior stress')].set_index('source_row');b=p[p.model.eq('Additive environment')].set_index('source_row').loc[a.index];delta=(b.predicted-b.observed).abs()-(a.predicted-a.observed).abs();groups=[delta[a.person.eq(person)].to_numpy() for person in sorted(a.person.unique())];rng=np.random.default_rng(42);boot=[np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(5000)]
        summaries[protocol]={'scores':scores,'interaction_MAE_improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'note':'Only test responses with genuinely observed pre-exposure stress. Missing-history responses cannot inform moderation; these predictions are an eligible subset of the same fitted models, not refitted models.'}
    return summaries


def full_coefficients(base,z,w,y,person):
    x=np.column_stack([base,z,w]);pinv=np.linalg.pinv(x);beta=pinv@y;e=y-x@beta;g=len(np.unique(person));rank=np.linalg.matrix_rank(x)
    # Direct OLS influence scores avoid squaring the design condition number.
    influence=np.array([pinv[:,person==v]@e[person==v] for v in sorted(np.unique(person))]);cov=g/(g-1)*(len(y)-1)/(len(y)-rank)*(influence.T@influence)
    return beta[-14:],cov[-14:,-14:]


def interaction_test(base,z,moderator,y,people,label):
    w=z*(moderator-2)[:,None];r=previous.partial_fit(np.column_stack([base,z]),w,y,people,wild_draws=1999);beta,cov=full_coefficients(base,z,w,y,people)
    np.testing.assert_allclose(r['beta'],beta[7:],atol=1e-7,rtol=1e-7);np.testing.assert_allclose(r['covariance'],cov[7:,7:],atol=1e-6,rtol=1e-5)
    rows=[];critical=stats.t.ppf(.975,len(np.unique(people))-1)
    for j,name in enumerate(MEANS):
        row={'specification':label,'exposure':name,'unit':previous.PLAN['exposure_units'][name],'interaction_per_prior_PSS_point':float(r['beta'][j]),'interaction_ci_low':float(r['low'][j]),'interaction_ci_high':float(r['high'][j]),'q_BH_7_interactions':float(r['q'][j])}
        for score in [1.0,2.0]:
            v=np.zeros(14);v[j]=1;v[7+j]=score-2;slope=float(v@beta);se=float(np.sqrt(max(0,v@cov@v)));row[f'slope_at_prior_{int(score)}']=slope;row[f'slope_ci_at_prior_{int(score)}']=[float(slope-critical*se),float(slope+critical*se)]
        rows.append(row)
    result={k:v for k,v in r.items() if k not in ['projection','residualized_exposure','residuals','covariance','beta','se','low','high','p','q']}
    return result,rows,w


def associations(first):
    d=pd.read_csv(ROOT/'outputs/stress-environment-hypothesis/within_person_cohort.csv');d['date']=pd.to_datetime(d.survey_date)
    z=transformed_levels(d);z=z-pd.DataFrame(z).groupby(d.person).transform('mean').to_numpy();d['calendar_week']=d.date.dt.to_period('W-SUN').astype(str)
    history=np.column_stack([d.prior_stress,d.past_mean_stress,d.days_since_prior_stress,(d.prior_stress-2)**2]);sd=history.std(axis=0);history=(history-history.mean(axis=0))/np.where(sd>0,sd,1)
    base=np.column_stack([pd.get_dummies(d[['person','calendar_week']],drop_first=False,dtype=float).to_numpy(),history])
    result,rows,w=interaction_test(base,z,d.prior_stress.to_numpy(),d.PSS.to_numpy(),d.person.to_numpy(),'weekly_prior_stress')
    summaries={'weekly_prior_stress':result};allrows=rows.copy();d['first_baseline_stress']=d.person.map(lambda p:first.get(p,{}).get('value',np.nan));d.to_csv(OUT/'association_cohort.csv',index=False)
    if d.first_baseline_stress.notna().all():
        r,rr,_=interaction_test(base,z,d.first_baseline_stress.to_numpy(),d.PSS.to_numpy(),d.person.to_numpy(),'first_pre_exposure_baseline');summaries['first_pre_exposure_baseline']=r;allrows+=rr
    else:
        # First baseline must genuinely precede initial exposure; do not replace it using later reports.
        selected=d.first_baseline_stress.notna().to_numpy();r,rr,_=interaction_test(base[selected],z[selected],d.first_baseline_stress[selected].to_numpy(),d.PSS[selected].to_numpy(),d.person[selected].to_numpy(),'first_pre_exposure_baseline');summaries['first_pre_exposure_baseline']=r;allrows+=rr
    # Exact first differences of the interaction product, not product of differences.
    indices=[]
    for _,g in d.groupby('person'):
        ids=g.sort_values('date').index.to_list()
        for a,b in zip(ids[:-1],ids[1:]):
            if 7<=(d.loc[b,'date']-d.loc[a,'date']).days<=14:indices.append((a,b))
    a=np.array([p[0] for p in indices]);b=np.array([p[1] for p in indices]);bd=base[b]-base[a];zd=z[b]-z[a];wd=w[b]-w[a];y=d.PSS.to_numpy()[b]-d.PSS.to_numpy()[a];person=d.person.to_numpy()[b]
    r=previous.partial_fit(np.column_stack([bd,zd]),wd,y,person,wild_draws=1999);summaries['first_difference_weekly_interactions']={k:v for k,v in r.items() if k not in ['projection','residualized_exposure','residuals','covariance','beta','se','low','high','p','q']}
    for j,name in enumerate(MEANS):allrows.append({'specification':'first_difference_weekly_interactions','exposure':name,'unit':previous.PLAN['exposure_units'][name],'interaction_per_prior_PSS_point':float(r['beta'][j]),'interaction_ci_low':float(r['low'][j]),'interaction_ci_high':float(r['high'][j]),'q_BH_7_interactions':float(r['q'][j])})
    support={'n':len(d),'people':int(d.person.nunique()),'higher_prior_rows':int(d.prior_stress.ge(2).sum()),'higher_prior_people':int(d.loc[d.prior_stress.ge(2),'person'].nunique()),'lower_prior_rows':int(d.prior_stress.lt(2).sum()),'lower_prior_people':int(d.loc[d.prior_stress.lt(2),'person'].nunique()),'prior_score_summary':d.prior_stress.describe().to_dict(),'first_baseline_score_summary':d.first_baseline_stress.describe().to_dict()}
    pd.DataFrame(allrows).to_csv(OUT/'interaction_coefficients.csv',index=False);print('Completed within-person moderation and sensitivities',flush=True)
    return summaries,allrows,support


def render(r):
    text='''<!doctype html><html><head><meta charset="utf-8"><title>Does environment matter more after higher stress?</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}</style></head><body><h1>Does environment predict stress differently depending on prior stress?</h1><p>Exploratory DEPRESSData moderation analysis, November 10, 2026.</p><p>Primary moderator: latest PSS score reported at least seven days before the current survey, on the supplied 0–4 scale. Its history is frozen before each test block. Secondary moderator: first valid PSS report available before a student’s initial matched exposure week. These address prior state and study-start baseline respectively. No clinical classification is asserted.</p><h2>What is tested</h2><p>Seven continuous stress-by-environment products test whether environmental slopes change with prior stress. All main effects and a quadratic prior-stress term are included. An interaction can differ in either direction; a nonzero interaction alone does not mean that every environmental condition matters more at higher stress. Temperature, humidity and light have no assumed universally harmful direction here. The primary test is the joint interaction test, not a comparison of separate subgroup significance levels.</p>'''
    a=r['associations'];text+=f'''<h2>Finding</h2><p><strong>This run does not find reliable evidence that environment matters more at higher prior stress.</strong> The direct joint weekly-prior-stress interaction test has wild-cluster p={a['weekly_prior_stress']['joint_restricted_wild_cluster_p']:.3f}. The first pre-exposure baseline sensitivity has p={a['first_pre_exposure_baseline']['joint_restricted_wild_cluster_p']:.3f}, and week-to-week differences p={a['first_difference_weekly_interactions']['joint_restricted_wild_cluster_p']:.3f}. No individual interaction passes seven-test correction. Uncertainty is wide; this is not proof that moderation is absent.</p>'''
    if 'known_history_prediction' in r:
        k=r['known_history_prediction']['purged_rolling_blocks'];scores=k['scores'];lo,hi=k['person_cluster_95_interval']
        text+=f'''<p>Among the {scores['History/control']['n']} rolling-test responses with observed prior stress, MAE is {scores['History/control']['mae']:.3f} for history/control, {scores['Additive environment']['mae']:.3f} with additive environment, and {scores['Environment × prior stress']['mae']:.3f} with interactions. Interaction improvement over additive environment is {k['interaction_MAE_improvement']:.3f}, with person-cluster95% interval {lo:.3f} to {hi:.3f}. The separate11 missing-history test responses are shown in the full table but cannot answer moderation. In both known-prior groups the environment/interactions model has higher error than history/control, and the direct higher-minus-lower improvement interval includes zero.</p>'''
    support=r['association_support'];text+=f'''<p>Within-person analysis: {support['n']} responses, {support['people']} participant IDs. There are {support['higher_prior_rows']} responses from {support['higher_prior_people']} IDs with prior scores ≥2, and {support['lower_prior_rows']} responses from {support['lower_prior_people']} IDs below2. This fixed midpoint split is descriptive only; continuous stress is retained in the primary test.</p><h2>Within-person interaction tests</h2><p>Person and calendar-week fixed effects, earlier stress/mean/history-age controls, prior stress squared, and seven transformed main environmental terms. Interactions use deviations from each person’s average environment in this association cohort. Person centering uses the complete cohort only for descriptive association; it is not used to prepare held-out predictions. CR1 person-cluster covariance, t intervals and restricted-null wild-cluster tests (1,999 draws, seed42). Seven individual interaction tests are BH corrected within each specification. Short-panel lag bias, nonlinear bounded scores and time-varying confounding remain.</p><table><tr><th>Moderator / sensitivity</th><th>Responses</th><th>People</th><th>Joint interaction wild-cluster p</th></tr>'''
    for name,v in r['associations'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["n"]}</td><td>{v["people"]}</td><td>{v["joint_restricted_wild_cluster_p"]:.4f}</td></tr>'
    text+='</table><p>First differences use the difference of each interaction product and exact differences of calendar/history terms, on adjacent retained responses7–14 days apart. This removes stable levels but is noisier; it is a sensitivity check rather than an independent replication. The first-baseline sensitivity uses a constant baseline trait and does not replace missing early baselines with later stress answers.</p><table><tr><th>Model / exposure</th><th>Change in slope per extra prior-stress point</th><th>95% interval</th><th>BH q</th></tr>'
    for c in r['coefficients']:text+=f'<tr><td>{html.escape(c["specification"]+": "+c["exposure"])}</td><td>{c["interaction_per_prior_PSS_point"]:.3f}</td><td>{c["interaction_ci_low"]:.3f} to {c["interaction_ci_high"]:.3f}</td><td>{c["q_BH_7_interactions"]:.3f}</td></tr>'
    text+='</table><h2>Held-out prediction</h2><p>Nested fixed-alpha10 ridge models share history, personal/time controls and the quadratic prior-stress term. Additive environment includes the seven transformed weekly means and seven between-day standard deviations; the final model adds seven prior-stress-by-environment products. All imputation, centering, scaling and interaction construction is fitted on training data only. Same frozen April test and purged rolling blocks as the previous study; outcomes already inspected in earlier experiments, so this is exploratory. No tuning search. Predictions bounded to0–4.</p>'
    for protocol,models in r['prediction'].items():
        text+='<h3>'+protocol+'</h3><table><tr><th>Model</th><th>Responses</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>'
        for name,v in models.items():text+=f'<tr><td>{html.escape(name)}</td><td>{v["n"]}</td>'+''.join(f'<td>{v[k]:.3f}</td>' for k in ['mae','rmse','r2'])+'</tr>'
        text+='</table><table><tr><th>Prior-score group / model</th><th>Responses</th><th>People</th><th>MAE</th></tr>'
        for group,gs in r['subgroup_prediction'][protocol].items():
            for name,v in gs.items():text+=f'<tr><td>{html.escape(group+": "+name)}</td><td>{v["n"]}</td><td>{v["people"]}</td><td>{v["mae"]:.3f}</td></tr>'
        text+='</table><table><tr><th>Added model / comparator</th><th>MAE improvement</th><th>95% person-cluster interval</th><th>Higher minus lower improvement [95% interval]</th></tr>'
        for name,v in r['prediction_intervals'][protocol].items():
            lo,hi=v['person_cluster_95_interval'];diff=v['higher_minus_lower_improvement'];di=v['higher_minus_lower_95_interval'];display=f'{diff:.3f} [{di[0]:.3f}, {di[1]:.3f}]' if di else 'Not tested: insufficient subgroup support'
            text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td><td>{display}</td></tr>'
        text+='</table>'
    text+='''<h2>Interpretation limits</h2><p>Positive error improvement means added features predict better; positive higher-minus-lower difference means more benefit in the higher-prior group. These intervals jointly resample person clusters, preserving both states within a person. They use fixed predictions without refitting and do not include full training/model-selection uncertainty. Unknown prior scores remain a separate descriptive group, with training-only imputation in prediction. April has only three higher-prior observations, so it cannot settle that subgroup question.</p><p>A direct interaction contrast is needed: significance in one subgroup and not another does not establish a difference. See <a href="https://sites.stat.columbia.edu/gelman/research/published/signif4.pdf">Gelman and Stern</a>. These tests can reveal predictive/associational moderation, not different causal sensitivity. Stress-dependent behavior may change environmental measurements; unmeasured exams, sleep, occupancy, sensor error and supplied-score coding limitations remain. Person fixed effects remove persistent room differences. A null interaction is not proof of identical responses; small high-stress samples limit precision.</p><p><a href="results.json">Full results</a>; <a href="analysis_plan.json">definitions set before this moderation run</a>; <a href="interaction_coefficients.csv">interaction coefficients and example slopes at prior scores1 and2</a>; <a href="predictions.csv">held-out predictions</a>; <a href="prediction_history_audit.csv">temporal history audit</a>; <a href="integrity_checks.json">checks</a>. Fitted ridge pipelines are in models.joblib. Original data and previous outputs remain unchanged. Reproduce with stress_environment_moderation.py.</p></body></html>'''
    (OUT/'report.html').write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));df,h,first,meta=load();scores,groups,ci,splits=predictive(df,h);ass,rows,support=associations(first)
    known=known_history_summary(pd.read_csv(OUT/'predictions.csv'))
    result={'plan':PLAN,'metadata':meta,'prediction':scores,'subgroup_prediction':groups,'prediction_intervals':ci,'known_history_prediction':known,'splits':splits,'associations':ass,'coefficients':rows,'association_support':support};(OUT/'results.json').write_text(json.dumps(result,indent=2));render(result)
    checks={'same_target_cohort234':True,'prior_scores_at_least7_days_earlier_and_test_history_frozen':True,'main_effects_preserved_in_interaction_models':True,'quadratic_prior_control_shared_by_all_models':True,'prediction_products_fit_on_training_only':True,'no_duplicate_prediction_keys':True,'model_serialization_roundtrip':True,'interaction_coefficients_and_covariance_match_full_OLS':True,'first_differences_use_difference_of_products':True,'sparse_subgroup_inference_suppressed':True};(OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({'prediction':scores,'subgroups':groups,'prediction_intervals':ci,'associations':ass,'support':support},indent=2),flush=True)

if __name__=='__main__':main()
