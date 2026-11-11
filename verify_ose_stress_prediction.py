"""Independent source-history, chronological and metric checks for later OSE."""
import json
import numpy as np
import pandas as pd
import ose_stress_prediction as analysis
OUT=analysis.OUT
sources={}
for prefix,filename,score in [('ose','OSE_Computer_Internert.csv','OSE'),('stress','PANAS_PSS.csv','PSS')]:
    raw=pd.read_csv(next(analysis.regression.SOURCE.rglob(filename)));raw['source_row']=raw.index+2;raw['person']=raw.ID.map(analysis.canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce');raw[score]=pd.to_numeric(raw[score],errors='coerce')
    d=raw[raw.person.notna()&raw.date.notna()&np.isfinite(raw[score])&raw[score].ge(0)]
    if prefix=='stress':d=d[d.PSS.le(4)]
    counts=d.groupby(['person','date'])[score].nunique();valid=set(counts[counts.eq(1)].index);d=d[[tuple(x) in valid for x in d[['person','date']].to_numpy()]].drop_duplicates(['person','date'])
    sources[prefix]={'score':score,'by_id':{p:g.sort_values('date') for p,g in d.groupby('person')},'by_row':d.set_index('source_row')}
audit=pd.read_csv(OUT/'history_availability_audit.csv',float_precision='round_trip');checked=set()
for _,r in audit.iterrows():
    key=(r.person,r.date,r.cutoff_exclusive if r.role=='test' else None)
    if key in checked:continue
    checked.add(key);date=pd.Timestamp(r.date);target=sources['ose']['by_row'].loc[r.source_row];np.testing.assert_allclose(r.OSE,target.OSE,atol=1e-12);assert target.person==r.person
    for prefix in ['ose','stress']:
        score=sources[prefix]['score'];old=sources[prefix]['by_id'][r.person];old=old[old.date.lt(date-pd.Timedelta(days=7))]
        if r.role=='test':old=old[old.date.lt(pd.Timestamp(r.cutoff_exclusive))]
        assert len(old)==r[prefix+'_count'] and len(old)>=(2 if prefix=='ose' else 1)
        assert set(map(int,r[prefix+'_history_source_rows'].split(',')))==set(old.source_row)
        values=old[score].to_numpy();mean=sum(values)/len(values);sd=np.sqrt(sum((values-mean)**2)/len(values));last=old.iloc[-1]
        for field,expected in [('mean',mean),('last',values[-1]),('last_deviation',values[-1]-mean),('sd',sd),('age_days',(date-last.date).days)]:np.testing.assert_allclose(r[prefix+'_'+field],expected,atol=1e-10)
        if len(old)>=2:np.testing.assert_allclose(r[prefix+'_last_change'],values[-1]-values[-2],atol=1e-12)
        else:assert pd.isna(r[prefix+'_last_change'])
        assert last.date<date-pd.Timedelta(days=7)
        if prefix=='ose':assert r.source_row not in set(old.source_row)
p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');result=json.loads((OUT/'results.json').read_text())
for protocol,pp in p.groupby('protocol'):
    sets=[set(g.source_row) for _,g in pp.groupby('model')];assert all(s==sets[0] for s in sets)
    refs=audit[audit.protocol.eq(protocol)&audit.role.eq('test')].set_index('source_row');assert refs.index.is_unique
    for name,g in pp.groupby('model'):
        assert g.source_row.is_unique;target=refs.loc[g.source_row];np.testing.assert_array_equal(g.observed.to_numpy(),target.OSE.to_numpy())
        v=result['scores'][protocol][name];np.testing.assert_allclose(v['mae'],np.mean(abs(g.predicted-g.observed)),atol=1e-12);np.testing.assert_allclose(v['rmse'],np.sqrt(np.mean((g.predicted-g.observed)**2)),atol=1e-12)
        if name=='Past mean OSE':np.testing.assert_allclose(g.predicted.to_numpy(),target.ose_mean.to_numpy(),atol=1e-12)
        if name=='Last OSE':np.testing.assert_array_equal(g.predicted.to_numpy(),target.ose_last.to_numpy())
    for name,v in result['comparisons'][protocol].items():
        added,baseline=name.split(' versus ');a=pp[pp.model.eq(added)].set_index('source_row');b=pp[pp.model.eq(baseline)].set_index('source_row').loc[a.index]
        np.testing.assert_allclose(v['mae_improvement'],np.mean(abs(b.observed-b.predicted)-abs(a.observed-a.predicted)),atol=1e-12)
checks={'OSE_and_PSS_cleaning_reconstructed_from_sources':True,'all_distinct_source_histories_and14_history_features_recomputed':True,'no_target_or_same_week_reports_enter_history':True,'all_test_histories_strictly_before_block_training_cutoff':True,'source_targets_and_person_matches_reproduced':True,'all_models_share_test_rows':True,'all_MAE_RMSE_and_paired_improvement_means_recomputed':True,'personalized_baseline_predictions_reproduced':True,'audit_rows':len(audit),'distinct_target_history_keys':len(checked)}
(OUT/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
