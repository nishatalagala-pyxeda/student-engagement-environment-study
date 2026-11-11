"""Stress history vs bedroom environment: prospective ablations and within-person associations.
All findings exploratory. This observational design does not identify a causal environmental effect.
"""
from pathlib import Path
import os,json,hashlib,html,sys
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
from scipy import stats
import scipy,sklearn,joblib
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
import depress_weekly_stress as original
from assess_depressdata import canonical
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/stress-environment-hypothesis'
HISTORY=['prior_stress','past_mean_stress','prior_report_count','days_since_prior_stress']
PLAN={
 'question':'Does measured bedroom environment have an effect on stress? Causal effect not identified; evaluate incremental prediction and within-person associations.',
 'predictive_hypothesis':'Adding measured environment to history plus identical personal/time controls reduces held-out MAE.',
 'primary_comparison':'History + controls | forest versus History + controls + environment | forest. Same family/settings, observations, split and admissible past labels.',
 'history_rule':'All valid nonconflicting PSS-only historical rows. Each prior report at least7 days before target report. For a test block, also strictly before training cutoff. No test block stress reports used, even for later test weeks.',
 'forward_test':'Previously used test: April1,2021 onward; training dates before March25. Reused exploratory test, not a new untouched validation sample.',
 'rolling_test':'Nonoverlapping14-day blocks starting June1,2020, expanded forward across study period. Train strictly earlier than block start minus7 days; require40 matched training responses and10 participant IDs. Each test response scored once. This avoids relying only on Spring.',
 'associations':'Person and calendar-week fixed effects; seven transformed weekly environmental means jointly. Primary associations adjusted for last and average pre-exposure PSS plus age of last report. Sensitivity without history. At least3 usable nonoverlapping weeks per person.',
 'exposure_units':{'temperature_C_week_mean':'1 degree C','humidity_week_mean':'10 percentage points','co2_week_mean':'100 reported units (normally ppm; calibration not verified)','voc_week_mean':'doubling of1+reported VOC','pm25_week_mean':'doubling of1+reported PM2.5','light_week_mean':'doubling of1+reported light','noise_week_mean':'5 reported noise units'},
 'inference':'Person-cluster CR1 covariance, t intervals withG-1 degrees of freedom, BH correction over7 environment tests within each specification. Joint test uses restricted-null Rademacher wild-person-cluster bootstrap,1999 draws, seed42. No claim of absence from nonsignificance.',
 'limits':'Unmeasured time-varying confounding (exams/sleep/occupancy), reverse causation within exposure week, exposure measurement error, short-panel lag/FE bias, no randomization, possible cross-cohort identity ambiguity. FE removes persistent between-person environmental differences; history adjustment can absorb prior environmental effects. Neither null predictions nor associations establish no causal effect.'}


def load():
    df=pd.read_csv(ROOT/'outputs/depress-weekly-stress/modeling_data.csv');df['date']=pd.to_datetime(df.survey_date)
    assert len(df)==234 and df.person.nunique()==38
    source=next(original.SOURCE.rglob('PANAS_PSS.csv'));raw=pd.read_csv(source);raw['person']=raw.ID.map(canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce');raw['source_row']=raw.index+2
    h=raw[raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4)].copy()
    conflicts=h.groupby(['person','date']).PSS.transform('nunique').gt(1);excluded=h[conflicts].copy();h=h[~conflicts].drop_duplicates(['person','date']).sort_values(['person','date']).reset_index(drop=True)
    excluded.to_csv(OUT/'excluded_conflicting_history.csv',index=False)
    # Check original modeling outcomes survive target-only cleaning; no disagreement is overwritten.
    target=h.set_index(['person','date']).PSS.reindex(pd.MultiIndex.from_frame(df[['person','date']]))
    np.testing.assert_allclose(target.to_numpy(),df.PSS.to_numpy(),rtol=0,atol=0)
    metadata={'n':len(df),'people':int(df.person.nunique()),'history_rows':len(h),'history_people':int(h.person.nunique()),'excluded_conflicting_history_rows':len(excluded),'raw_history_missing_or_invalid_rows':len(raw)-len(raw[raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4)]),'history_source':str(source),'history_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'prepared_input_sha256':hashlib.sha256((ROOT/'outputs/depress-weekly-stress/modeling_data.csv').read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'packages':{m.__name__:m.__version__ for m in [np,pd,scipy,sklearn,joblib]},'python':sys.version}
    return df,h,metadata


def add_history(df,history,cutoff=None):
    d=df.copy();records=[]
    for _,row in d.iterrows():
        eligible=history[history.person.eq(row.person)&history.date.le(row.date-pd.Timedelta(days=7))]
        if cutoff is not None:eligible=eligible[eligible.date.lt(cutoff)]
        eligible=eligible.sort_values('date')
        if len(eligible):
            prior=eligible.iloc[-1];r={'prior_stress':float(prior.PSS),'past_mean_stress':float(eligible.PSS.mean()),'prior_report_count':len(eligible),'days_since_prior_stress':(row.date-prior.date).days,'latest_history_date':prior.date}
            assert r['days_since_prior_stress']>=7 and (cutoff is None or prior.date<cutoff)
        else:r={'prior_stress':np.nan,'past_mean_stress':np.nan,'prior_report_count':0,'days_since_prior_stress':np.nan,'latest_history_date':pd.NaT}
        records.append(r)
    for c in records[0]:d[c]=[r[c] for r in records]
    return d


def scopes_for():
    m=json.loads((ROOT/'outputs/depress-weekly-stress/metrics.json').read_text())['metadata'];env=m['environment_features'];base=m['scope_features']['Personal/time'];cats=['recorded_sex','income_category','household_size_category','bedroom_count_category','study_cohort'];num=[c for c in base if c not in cats]
    return {'Controls':(num,cats),'Controls + environment':(num+env,cats),'History + controls':(num+HISTORY,cats),'History + controls + environment':(num+HISTORY+env,cats)}


def get_splits(df,protocol):
    if protocol=='frozen_April_test':
        cutoff=pd.Timestamp('2021-03-25');start=pd.Timestamp('2021-04-01');train=np.flatnonzero(df.date.lt(cutoff));test=np.flatnonzero(df.date.ge(start));yield 'April',cutoff,train,test
    elif protocol=='purged_rolling_blocks':
        for start in pd.date_range('2020-06-01',df.date.max(),freq='14D'):
            cutoff=start-pd.Timedelta(days=7);train=np.flatnonzero(df.date.lt(cutoff));test=np.flatnonzero(df.date.ge(start)&df.date.lt(start+pd.Timedelta(days=14)))
            if len(train)>=40 and df.iloc[train].person.nunique()>=10 and len(test):yield str(start.date()),cutoff,train,test
    else:raise ValueError(protocol)


def predict_evaluation(df,history,scopes):
    specs={}
    for scope,(nums,cats) in scopes.items():
        for forest,family in [(False,'ridge'),(True,'forest')]:specs[scope+' | '+family]=(nums+cats,original.build(nums,cats,forest))
    specs['Mean baseline']=(['annual_sin'],DummyRegressor(strategy='mean'));specs['Median baseline']=(['annual_sin'],DummyRegressor(strategy='median'))
    records=[];audits=[];splitinfo=[]
    for protocol in ['frozen_April_test','purged_rolling_blocks']:
        seen=set()
        for block,cutoff,tr,te in get_splits(df,protocol):
            training=add_history(df.iloc[tr].reset_index(drop=True),history);test=add_history(df.iloc[te].reset_index(drop=True),history,cutoff)
            assert training.date.max()<cutoff and training.date.max()<=test.date.min()-pd.Timedelta(days=7)
            assert set(training.source_row).isdisjoint(test.source_row) and seen.isdisjoint(test.source_row);seen.update(test.source_row)
            assert training.date.max()<=test.date.min()-pd.Timedelta(days=7)
            splitinfo.append({'protocol':protocol,'block':block,'history_cutoff_exclusive':str(cutoff.date()),'train_n':len(training),'train_people':int(training.person.nunique()),'train_latest':str(training.date.max().date()),'test_n':len(test),'test_first':str(test.date.min().date()),'test_last':str(test.date.max().date()),'test_cohorts':test.study_cohort.value_counts().to_dict()})
            for kind,frame in [('train',training),('test',test)]:
                for _,r in frame.iterrows():audits.append({'protocol':protocol,'block':block,'role':kind,'source_row':int(r.source_row),'person':r.person,'target_date':str(r.date.date()),'history_latest':str(r.latest_history_date.date()) if pd.notna(r.latest_history_date) else None,'history_cutoff_exclusive':str(cutoff.date()) if kind=='test' else None,**{c:r[c] for c in HISTORY}})
            predictions={}
            for name,(cols,template) in specs.items():
                assert not set(cols)&{'PSS','person','source_row','date','latest_history_date'}
                model=clone(template).fit(training[cols],training.PSS);predictions[name]=np.clip(model.predict(test[cols]),0,4)
            for label,col in [('Past mean stress baseline','past_mean_stress'),('Last stress baseline','prior_stress')]:predictions[label]=test[col].fillna(training.PSS.mean()).to_numpy(float)
            for name,p in predictions.items():
                for (_,r),value in zip(test.iterrows(),p):records.append({'protocol':protocol,'block':block,'model':name,'source_row':int(r.source_row),'person':r.person,'survey_date':str(r.date.date()),'observed':float(r.PSS),'predicted':float(value)})
        print('Completed prediction protocol',protocol,flush=True)
    pred=pd.DataFrame(records);pred.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'history_availability_audit.csv',index=False)
    assert not pred.duplicated(['protocol','model','source_row']).any()
    scores={}
    for (protocol,model),p in pred.groupby(['protocol','model'],sort=False):scores.setdefault(protocol,{})[model]=original.score(p.observed.to_numpy(),p.predicted.to_numpy())
    ci={}
    comparisons=[('History + controls | '+f,'History + controls + environment | '+f) for f in ['forest','ridge']]
    comparisons += [('Controls | forest','History + controls | forest'),('Controls | ridge','History + controls | ridge'),('Mean baseline','Past mean stress baseline'),('Past mean stress baseline','History + controls + environment | forest')]
    for protocol in pred.protocol.unique():
        ci[protocol]={};p=pred[pred.protocol.eq(protocol)]
        for baseline,augmented in comparisons:
            a=p[p.model.eq(augmented)].set_index('source_row');b=p[p.model.eq(baseline)].set_index('source_row').loc[a.index]
            delta=(b.predicted-b.observed).abs()-(a.predicted-a.observed).abs();groups=[delta[a.person.eq(v)].to_numpy() for v in sorted(a.person.unique())];rng=np.random.default_rng(42)
            boot=[np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(5000)]
            ci[protocol][augmented+' versus '+baseline]={'mae_improvement':float(delta.mean()),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'people':len(groups),'n':len(a),'interpretation':'Positive favors augmented model. Fixed paired prediction clusters, no refitting; not full training/selection uncertainty.'}
    # Full-data research fits use historically admissible features for each row.
    full=add_history(df,history);fitted={n:clone(t).fit(full[c],full.PSS) for n,(c,t) in specs.items()}
    payload={'models':fitted,'features':{n:c for n,(c,_) in specs.items()},'bounds':(0,4),'history_rule':PLAN['history_rule'],'plan':PLAN};joblib.dump(payload,OUT/'prediction_models.joblib')
    restored=joblib.load(OUT/'prediction_models.joblib');name='History + controls + environment | forest';np.testing.assert_allclose(restored['models'][name].predict(full[payload['features'][name]]),fitted[name].predict(full[payload['features'][name]]))
    return scores,ci,splitinfo


def bh_adjust(p):
    p=np.asarray(p);order=np.argsort(p);ranked=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];result=np.empty(len(p));result[order]=np.minimum(ranked,1);return result


def partial_fit(base,z,y,groups,wild_draws=1999):
    # FWL residualization; covariance equals the exposure block of full OLS CR1.
    projection=base@np.linalg.pinv(base);residualizer=np.eye(len(y))-projection
    zr=residualizer@z;yr=residualizer@y;k=z.shape[1];rank=int(np.linalg.matrix_rank(np.column_stack([base,z])))
    assert np.linalg.matrix_rank(zr)==k and rank<len(y)
    bread=np.linalg.pinv(zr.T@zr);beta=bread@zr.T@yr;resid=yr-zr@beta
    labels,group_index=np.unique(groups,return_inverse=True);g=len(labels);factor=g/(g-1)*(len(y)-1)/(len(y)-rank)
    def covariance(e):
        scores=np.zeros((g,k));np.add.at(scores,group_index,zr*e[:,None]);return factor*bread@(scores.T@scores)@bread
    cov=covariance(resid);se=np.sqrt(np.maximum(0,np.diag(cov)));t=beta/se;p=2*stats.t.sf(np.abs(t),df=g-1);critical=stats.t.ppf(.975,df=g-1)
    f=float(beta@np.linalg.pinv(cov)@beta/k);f_approx=float(stats.f.sf(f,k,g-1))
    rng=np.random.default_rng(42);bootstrap=[]
    for _ in range(wild_draws):
        sign=rng.choice([-1.0,1.0],size=g);yb=residualizer@(yr*sign[group_index]);b=bread@zr.T@yb;e=yb-zr@b;c=covariance(e);bootstrap.append(float(b@np.linalg.pinv(c)@b/k))
    return {'n':len(y),'people':g,'model_rank':rank,'residual_df':len(y)-rank,'joint_F':f,'joint_F_approx_p':f_approx,'joint_restricted_wild_cluster_p':float((1+np.greater_equal(bootstrap,f).sum())/(wild_draws+1)),'wild_draws':wild_draws,'beta':beta,'se':se,'low':beta-critical*se,'high':beta+critical*se,'p':p,'q':bh_adjust(p),'partial_r2_in_sample':float(1-resid@resid/(yr@yr)),'projection':projection,'residualized_exposure':zr,'residuals':resid,'covariance':cov}


def association(df,history):
    d=add_history(df,history);d=d[d.prior_stress.notna()].sort_values(['person','date']).copy()
    retained=[]
    for _,g in d.groupby('person'):
        last=None
        for idx,row in g.iterrows():
            if last is None or (row.date-last).days>=7:retained.append(idx);last=row.date
    d=d.loc[retained].copy();d=d[d.groupby('person').person.transform('size').ge(3)].reset_index(drop=True)
    d['calendar_week']=d.date.dt.to_period('W-SUN').astype(str);d.to_csv(OUT/'within_person_cohort.csv',index=False)
    names=list(PLAN['exposure_units']);z=np.column_stack([d[n].to_numpy(float) for n in names])
    z[:,1]/=10;z[:,2]/=100
    for j in [3,4,5]:z[:,j]=np.log2(1+z[:,j])
    z[:,6]/=5
    base=pd.get_dummies(d[['person','calendar_week']],drop_first=False,dtype=float).to_numpy();y=d.PSS.to_numpy(float)
    h=d[['prior_stress','past_mean_stress','days_since_prior_stress']].to_numpy(float);h=(h-h.mean(axis=0))/np.where(h.std(axis=0)>0,h.std(axis=0),1)
    outputs={};coefficients=[];integrity={}
    for name,b in [('person_week_FE_history_adjusted',np.column_stack([base,h])),('person_week_FE_without_history',base)]:
        r=partial_fit(b,z,y,d.person.to_numpy());
        # Verify residualization coefficients and covariance against independent full-design OLS.
        full=np.column_stack([b,z]);pinv=np.linalg.pinv(full);beta_full=pinv@y;np.testing.assert_allclose(r['beta'],beta_full[-len(names):],atol=1e-7,rtol=1e-7)
        assert np.max(np.abs(b.T@r['residuals']))<1e-6
        rank=int(np.linalg.matrix_rank(full));bread=np.linalg.pinv(full.T@full);scores=np.array([full[d.person.eq(person)].T@(y-full@beta_full)[d.person.eq(person)] for person in sorted(d.person.unique())]);g=d.person.nunique();cov_full=g/(g-1)*(len(d)-1)/(len(d)-rank)*bread@scores.T@scores@bread
        np.testing.assert_allclose(r['covariance'],cov_full[-len(names):,-len(names):],atol=1e-6,rtol=1e-5)
        integrity[name]={'full_OLS_coefficients_and_CR1_covariance_match_FWL':True,'nuisance_residual_orthogonality_verified':True}
        for j,feature in enumerate(names):coefficients.append({'specification':name,'exposure':feature,'unit':PLAN['exposure_units'][feature],'coefficient':float(r['beta'][j]),'person_cluster_se':float(r['se'][j]),'ci_low':float(r['low'][j]),'ci_high':float(r['high'][j]),'p_raw':float(r['p'][j]),'q_BH_7_tests':float(r['q'][j])})
        outputs[name]={k:v for k,v in r.items() if k not in ['projection','residualized_exposure','residuals','covariance','beta','se','low','high','p','q']}
        print('Completed association',name,'n',r['n'],'people',r['people'],flush=True)
    pd.DataFrame(coefficients).to_csv(OUT/'within_person_coefficients.csv',index=False)
    variation=[]
    for name in names:
        person_means=d.groupby('person')[name].transform('mean');variation.append({'exposure':name,'total_sd':float(d[name].std()),'within_person_sd':float((d[name]-person_means).std()),'between_person_mean_sd':float(d.groupby('person')[name].mean().std())})
    pd.DataFrame(variation).to_csv(OUT/'environment_variation.csv',index=False)
    return outputs,coefficients,integrity,{'n':len(d),'people':int(d.person.nunique()),'calendar_weeks':int(d.calendar_week.nunique()),'overlapping_weekly_target_windows_removed':len(df)-len(d),'note':'Count removed includes missing pre-exposure history, overlap pruning and fewer-than3-week persons. Dynamic lag/FE estimates can be biased in short panels; associations are exploratory.'}


def render(result):
    ev=result['prediction'];cis=result['prediction_intervals'];ass=result['associations'];cohort=result['association_cohort'];text='''<!doctype html><html><head><meta charset="utf-8"><title>Does environment affect stress?</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}</style></head><body><h1>Does bedroom environment affect stress?</h1><p>DEPRESSData hypothesis evaluation. Exploratory analysis, November10,2026.</p><p><strong>This study can test incremental prediction and within-person associations. It does not identify a causal effect of environment on stress.</strong> There is no randomized environmental intervention, participant presence is unknown, and important time-varying confounders such as exams and sleep are unmeasured in this analysis. Stress may also change behavior that changes room measurements. Limited predictive gain does not establish that environmental effects are absent.</p>'''
    rolling=ev['purged_rolling_blocks'];r0=rolling['History + controls | ridge']['mae'];r1=rolling['History + controls + environment | ridge']['mae'];f0=rolling['History + controls | forest']['mae'];f1=rolling['History + controls + environment | forest']['mae']
    text+=f'''<h2>Answer supported by these tests</h2><p><strong>Past stress is useful for prediction. Whether bedroom environment causally affects stress remains unresolved.</strong> In rolling tests over {rolling['Mean baseline']['n']} responses, adding environment increases ridge MAE from {r0:.3f} to {r1:.3f}, and forest MAE from {f0:.3f} to {f1:.3f}. Thus these environment features add no demonstrated predictive benefit beyond history and identical controls. Past-average stress alone scores {rolling['Past mean stress baseline']['mae']:.3f}, versus {rolling['Mean baseline']['mae']:.3f} for the training-wide mean.</p><p>The within-person/history-adjusted environmental joint test gives wild-cluster p={ass['person_week_FE_history_adjusted']['joint_restricted_wild_cluster_p']:.3f}, a possible association. Without history adjustment, p={ass['person_week_FE_without_history']['joint_restricted_wild_cluster_p']:.4f}. No individual exposure passes the seven-test BH adjustment at5%. This is model-dependent evidence, not a clean finding of either no environmental relationship or a causal effect.</p>'''
    sensitivity=result.get('sensitivity')
    if sensitivity:
        fd=sensitivity['first_differences'];lo=sensitivity['leave_one_person_out'];vr=lo['VOC_coefficient_range']
        text+=f'''<p>A follow-up first-difference check compares {fd['n']} week-to-week changes from {fd['people']} IDs, with differenced calendar-week indicators and prior-stress controls. The environmental joint wild-cluster p={fd['joint_restricted_wild_cluster_p']:.3f}; no individual exposure passes correction. The initial association is therefore not consistently confirmed across specifications. Leaving one person out keeps the positive VOC estimate between {vr[0]:.3f} and {vr[1]:.3f}, so it is not solely attributable to one student, but these omission checks are not independent confirmations.</p>'''
    text+='<h2>Prediction beyond past stress</h2><p>The same234 environmental-response observations are retained. Past stress uses all valid, PSS-only cleaned source reports, each at least7 days before the target report, so recall weeks do not overlap by date. Each validation block freezes the available history strictly before the training cutoff; no stress answers from that test block are used. This is stricter than allowing updates from earlier test answers. Raw history rows need not have environmental coverage. It also differs from the earlier matched-history baseline: including the full earlier survey history can change its mean and performance.</p><p>Compare identical fixed ridge(alpha10) and shallow forest(300 trees, depth3, leaf10) families, with or without14 environmental level/variability features. Personal/time controls are unchanged. All preprocessing is fitted inside training folds. These scores infer stress for the week with observed environment, not a forecast before that environment was observed. April is the already-inspected exploratory test. Rolling tests use disjoint14-day blocks and a7-day training gap, requiring40 training responses and10 IDs. Scored rolling blocks cover Fall and Spring; the Summer/early-Fall responses form the warm-up training data.</p>'
    for protocol,models in ev.items():
        text+='<h3>'+protocol+'</h3><table><tr><th>Model</th><th>MAE (lower better)</th><th>RMSE</th><th>R²</th></tr>'
        for name,v in models.items():text+='<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{v[k]:.3f}</td>' for k in ['mae','rmse','r2'])+'</tr>'
        text+='</table><table><tr><th>Added model / comparator</th><th>MAE improvement (positive better)</th><th>Person-cluster95% interval</th></tr>'
        for name,v in cis[protocol].items():
            lo,hi=v['person_cluster_95_interval'];text+=f'<tr><td>{html.escape(name)}</td><td>{v["mae_improvement"]:.3f}</td><td>{lo:.3f} to {hi:.3f}</td></tr>'
        text+='</table>'
    text+=f'''<h2>Within-person environmental associations</h2><p>The association cohort has{cohort['n']} responses from{cohort['people']} IDs over{cohort['calendar_weeks']} calendar weeks. At least3 retained, nonoverlapping survey weeks per person and pre-exposure stress history are required. Person fixed effects remove stable individual/room differences; calendar-week fixed effects remove shared weekly shifts. All7 environmental means enter jointly, with skewed VOC/PM/light log-transformed. History-adjusted and no-history specifications are both shown. The lag-adjusted short panel can have dynamic-panel bias, so it is a sensitivity-style association model, not a causal estimator.</p><p>Person-cluster CR1 t intervals account for repeated measurements within an ID. The joint restricted-null wild-cluster test flips residual signs by person(1999 draws). Individual tests receive Benjamini–Hochberg correction within each7-exposure model. Shared residual calendar/cohort dependence and model selection are not fully captured. Singleton calendar periods may carry little information. Fixed effects remove persistent environmental differences, so they cannot rule out long-term effects of consistently poor rooms.</p>'''
    for name,v in ass.items():text+=f'<p>{html.escape(name)}: joint wild-cluster p={v["joint_restricted_wild_cluster_p"]:.4f}; residual degrees of freedom={v["residual_df"]}. In-sample partial R²={v["partial_r2_in_sample"]:.3f} is descriptive and is not a validation score.</p>'
    text+='<table><tr><th>Model / exposure</th><th>Change in PSS per exposure unit</th><th>95% interval</th><th>BH q</th></tr>'
    for c in result['coefficients']:text+=f'<tr><td>{html.escape(c["specification"]+": "+c["exposure"])}<br>{html.escape(c["unit"])}</td><td>{c["coefficient"]:.3f}</td><td>{c["ci_low"]:.3f} to {c["ci_high"]:.3f}</td><td>{c["q_BH_7_tests"]:.3f}</td></tr>'
    text+='''</table><h2>What the answer can mean</h2><p>Evidence of additional prediction or a within-person association would justify further study, not prove causation. Lack of evidence does not show environment has no effect: exposure error, insufficient variation, few participants, nonlinear/short-lived effects, and power limitations remain. Stress history may encode persistent environmental conditions or mediate earlier environmental effects; controlling it can remove part of an environmental pathway. “History predicts better” and “environment does not affect stress” are different claims.</p><p>The supplied aggregate PSS score is modeled on0–4. Raw item scoring cannot be independently validated. Sensor IDs are inferred one-to-one matches; F135 remains unresolved/excluded. The bedroom is not verified exposure location. No clinical cutoff or causal effect size is asserted. This follow-up was motivated by prior inspected results and is exploratory.</p><h2>Methods and files</h2><p>Causal design limitations follow <a href="https://www.bmj.com/content/378/bmj-2022-071108">Hernán et al., target-trial principles</a>. Cluster inference follows <a href="https://www.nber.org/papers/t0344">Cameron, Gelbach and Miller</a>. Time validation uses calendar gaps, consistent with <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html">time-series validation principles</a>; custom folds handle irregular panel dates.</p><p><a href="results.json">All results</a>; <a href="analysis_plan.json">analysis definitions fixed before this run</a>; <a href="predictions.csv">held-out predictions</a>; <a href="history_availability_audit.csv">history cutoff audit</a>; <a href="within_person_coefficients.csv">association coefficients</a>; <a href="within_person_cohort.csv">association cohort</a>; <a href="integrity_checks.json">integrity checks</a>. Research model payloads are saved locally. No source data or earlier results are overwritten. Reproduce with stress_environment_hypothesis.py.</p></body></html>'''
    (OUT/'report.html').write_text(text)


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2));df,history,meta=load();scopes=scopes_for()
    print('History cleaned',meta['history_rows'],'reports; matched targets',len(df),flush=True)
    prediction,ci,splits=predict_evaluation(df,history,scopes);ass,coefficients,integrity,cohort=association(df,history)
    checks={'target_values_match_original_PSS_only_cleaning':True,'all_prior_reports_at_least7_days_earlier':True,'test_histories_frozen_before_training_cutoff':True,'no_duplicate_test_predictions_within_protocol':True,'no_future_training_records_or_exposure_overlap_at_test_boundary':True,'same_family_features_splits_and_rows_for_environment_ablation':True,'serialization_roundtrip_passed':True,'association_checks':integrity}
    (OUT/'integrity_checks.json').write_text(json.dumps(checks,indent=2));result={'plan':PLAN,'metadata':meta,'prediction':prediction,'prediction_intervals':ci,'splits':splits,'associations':ass,'coefficients':coefficients,'association_cohort':cohort};(OUT/'results.json').write_text(json.dumps(result,indent=2));render(result)
    print(json.dumps({'prediction':prediction,'prediction_intervals':ci,'associations':ass,'association_cohort':cohort},indent=2),flush=True)

if __name__=='__main__':main()
