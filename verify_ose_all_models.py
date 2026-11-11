"""Chronology, metrics, baseline equivalence and portable-model checks."""
import json,hashlib
import numpy as np
import pandas as pd
import joblib
from ose_model_transforms import FacialReducer
import ose_all_models as models
OUT=models.OUT;schema=json.loads((OUT/'feature_schema.json').read_text());d=pd.read_csv(OUT/'modeling_features.csv',float_precision='round_trip',dtype={c:str for c in schema['categorical_features']});d.date=pd.to_datetime(d.date)
r=json.loads((OUT/'results.json').read_text());p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');old=pd.read_csv(models.ROOT/'outputs/ose-stress-prediction/predictions.csv',float_precision='round_trip');test=pd.read_csv(OUT/'test_features_audit.csv',float_precision='round_trip')
# Reconstruct every test questionnaire feature from independently source-checked cleaned tables.
stores={}
for tag,meta in schema['source_metadata'].items():
    if tag in ['demographics','diary']:continue
    raw=pd.read_csv(OUT/f'cleaned_{tag}.csv',float_precision='round_trip');raw.date=pd.to_datetime(raw.date)
    for person,g in raw.groupby('person'):
        for col in meta['columns']:
            v=g[g[col].notna()].sort_values('date');stores[(tag,person,col)]=(v.date.to_numpy(),v[col].to_numpy())
expected=[];columns=schema['episodic_features']
for _,row in test.iterrows():
    bound=min(pd.Timestamp(row.date)-pd.Timedelta(days=7),pd.Timestamp(row.cutoff_exclusive));rec={}
    for tag,meta in schema['source_metadata'].items():
        if tag in ['demographics','diary']:continue
        for col in meta['columns']:
            dates,v=stores.get((tag,row.person,col),(np.array([],dtype='datetime64[ns]'),np.array([])));n=np.searchsorted(dates,np.datetime64(bound,'ns'),side='left');key=tag+'_'+models.features.slug(col)
            if n:values=v[:n];mu=sum(values)/n;vals=[values[-1],mu,np.sqrt(sum((values-mu)**2)/n),(pd.Timestamp(row.date)-pd.Timestamp(dates[n-1])).days];assert pd.Timestamp(dates[n-1])<bound
            else:vals=[np.nan]*4
            rec.update({key+'_'+suffix:value for suffix,value in zip(['last','mean','sd','age_days'],vals)})
    expected.append([rec[c] for c in columns])
np.testing.assert_allclose(test[columns].to_numpy(float),np.array(expected),equal_nan=True,atol=1e-9,rtol=1e-9)
prior_audit=pd.read_csv(models.ROOT/'outputs/ose-stress-prediction/history_availability_audit.csv',float_precision='round_trip');prior_audit=prior_audit[prior_audit.role.eq('test')].set_index(['protocol','source_row']);idx=pd.MultiIndex.from_frame(test[['protocol','source_row']]);ref=prior_audit.loc[idx]
np.testing.assert_allclose(test[models.core.OSE_FEATURES+models.core.STRESS_FEATURES].to_numpy(float),ref[models.core.OSE_FEATURES+models.core.STRESS_FEATURES].to_numpy(float),equal_nan=True,atol=1e-12)
for protocol,pp in p.groupby('protocol'):
    sets=[set(g.source_row) for _,g in pp.groupby('model')];assert all(s==sets[0] for s in sets)
    for name,g in pp.groupby('model'):
        assert g.source_row.is_unique;targets=d.set_index('source_row').loc[g.source_row];np.testing.assert_array_equal(g.observed.to_numpy(),targets.OSE.to_numpy());v=r['scores'][protocol][name]
        np.testing.assert_allclose(v['mae'],np.mean(abs(g.predicted-g.observed)),atol=1e-12);np.testing.assert_allclose(v['rmse'],np.sqrt(np.mean((g.predicted-g.observed)**2)),atol=1e-12)
        if name in ['Last OSE','Past mean OSE','Population mean','Population median','Past OSE | ridge','Past OSE | forest']:
            original=old[old.protocol.eq(protocol)&old.model.eq(name)].set_index('source_row').loc[g.source_row];np.testing.assert_array_equal(g.predicted.to_numpy(),original.predicted.to_numpy())
    for name,v in r['comparisons'][protocol].items():
        added,baseline=name.split(' versus ');a=pp[pp.model.eq(added)].set_index('source_row');b=pp[pp.model.eq(baseline)].set_index('source_row').loc[a.index];np.testing.assert_allclose(v['mae_improvement'],np.mean(abs(b.observed-b.predicted)-abs(a.observed-a.predicted)),atol=1e-12)
print('Frozen histories, target rows, baselines and all metrics checked',flush=True)
# Recover any first-run __main__ transformer reference, then save with its portable module.
payload=joblib.load(OUT/'models.joblib');train=d[d.eligible];before={n:v['model'].predict(train[v['features']]) for n,v in payload['models'].items()}
joblib.dump(payload,OUT/'models.portable.joblib');restored=joblib.load(OUT/'models.portable.joblib')
for name,v in restored['models'].items():np.testing.assert_allclose(before[name],v['model'].predict(train[v['features']]),atol=1e-10)
(OUT/'models.portable.joblib').replace(OUT/'models.joblib')
for name,v in restored['models'].items():
    face=v['model'].steps[0][1].named_transformers_.get('facial')
    if face is None:continue
    x=train[schema['face_features']];np.testing.assert_allclose(face.imputer_.statistics_,x.median().fillna(0).to_numpy(),atol=1e-10)
    stats=face.imputer_.statistics_.copy();components=face.reducer_.components_.copy() if face.reducer_ is not None else None
    probe=x.iloc[:3].fillna(100000.).copy();face.transform(probe);np.testing.assert_array_equal(stats,face.imputer_.statistics_)
    if components is not None:np.testing.assert_array_equal(components,face.reducer_.components_)
print('Portable serialization and train-fitted facial preprocessing checked',flush=True)
# Report actual physical-sensor coverage separately from satisfaction questionnaires.
availability=pd.read_csv(OUT/'test_modality_coverage.csv');coverage_details={}
for protocol,a in availability.groupby('protocol'):
    data=d.set_index('source_row').loc[a.source_row];tf=test[test.protocol.eq(protocol)].set_index('source_row').loc[a.source_row]
    checks={k:a[k].to_numpy(bool) for k in schema['groups']}
    checks['physical_sensors']=data[[c for c in d if c.startswith('env_')]].notna().any(axis=1).to_numpy()
    checks['environment_satisfaction']=tf[[c for c in columns if c.startswith('satisfaction_')]].notna().any(axis=1).to_numpy()
    coverage_details[protocol]={k:{'reports':int(mask.sum()),'ids':int(a.loc[mask,'person'].nunique())} for k,mask in checks.items()}
for k,cols in schema['groups'].items():
    take=[c for c in cols if c!='stress_count' and (k!='demographics_time' or c.startswith('demo_'))];mask=d[take].notna().any(axis=1);schema['coverage'][k]['reports_any_feature']=int(mask.sum());schema['coverage'][k]['ids_any_feature']=int(d.loc[mask,'person'].nunique())
(OUT/'feature_schema.json').write_text(json.dumps(schema,indent=2));r['feature_schema_sha256']=hashlib.sha256((OUT/'feature_schema.json').read_bytes()).hexdigest();r['coverage']=schema['coverage'];r['actual_test_coverage']=coverage_details
r['archive_metadata']={k:json.loads((OUT/f'{k}_archive_metadata.json').read_text()) for k in ['fitbit','facial']}
for v in r['archive_metadata'].values():v.pop('features',None)
for protocol,pp in p.groupby('protocol'):
    for family in ['ridge','forest']:
        a='All without past OSE | '+family;b='Population mean';r['comparisons'][protocol][a+' versus '+b]=models.paired(pp[pp.model.eq(a)],pp[pp.model.eq(b)])
(OUT/'results.json').write_text(json.dumps(r,indent=2));models.render(r)
checks={'all_test_episodic_values_recomputed_with_frozen_cutoffs':True,'past_OSE_and_stress_features_match_prior_verified_test_histories':True,'all_models_share_test_rows_and_source_targets':True,'all_MAE_RMSE_and_paired_improvements_recomputed':True,'previous_baseline_predictions_reproduce_exactly':True,'all_saved_models_load_portably_and_reproduce_predictions':True,'facial_imputation_statistics_match_training_data':True,'test_transform_cannot_change_fitted_facial_parameters':True,'actual_sensor_coverage_separated_from_satisfaction_reports':True,'test_records_in_feature_audit':len(test)}
(OUT/'model_independent_checks.json').write_text(json.dumps(checks,indent=2));combined={'features':json.loads((OUT/'feature_independent_checks.json').read_text()),'models':checks};(OUT/'independent_checks.json').write_text(json.dumps(combined,indent=2));print(json.dumps(checks,indent=2))
