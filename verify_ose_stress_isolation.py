"""Verify stress-only predictions and descriptive paired-change calculations."""
import json,hashlib
import joblib
import numpy as np
import pandas as pd
import ose_stress_isolation as study

out=study.OUT;r=json.loads((out/'results.json').read_text());d,schema,_,_=study.current.load();indexed=d.set_index('source_row',drop=False)
p=pd.read_csv(out/'predictions.csv',float_precision='round_trip',dtype={'student_group':str});m=pd.read_csv(study.current.OUT/'outer_membership.csv',dtype={'student_group':str})
previous=pd.read_csv(study.ROOT/'outputs/ose-unseen-with-history/predictions.csv',float_precision='round_trip');previous=previous[previous.protocol.eq('Previous eligible subset')]
for fold,mm in m.groupby('fold'):
    train=indexed.loc[mm.loc[mm.role.eq('train'),'source_row']];test=indexed.loc[mm.loc[mm.role.eq('test'),'source_row']]
    assert set(train.person).isdisjoint(test.person) and set(train.student_group).isdisjoint(test.student_group)
    models=joblib.load(out/f'fold_{fold}.joblib')
    for name,v in models.items():
        cols=v['features'];assert cols==r['scopes'][name.split(' | ')[0]]
        assert not set(cols)&{'OSE','OSE_change','person','student_group','source_row','date','eligible'}
        pp=p[p.fold.eq(fold)&p.model.eq(name)].set_index('source_row').loc[test.source_row]
        np.testing.assert_allclose(pp.predicted,v['model'].predict(test[cols]),atol=1e-10)
        imputer=v['model'].steps[0][1].named_transformers_['numeric'].steps[0][1]
        np.testing.assert_allclose(imputer.statistics_,train[cols].median().fillna(0),atol=1e-10)
for name,g in p.groupby('model'):
    assert g.source_row.is_unique and set(g.source_row)==set(d.loc[d.eligible,'source_row'])
    np.testing.assert_array_equal(g.observed,indexed.loc[g.source_row].OSE)
    if name in ['Past OSE only | Ridge','Past OSE only | Random forest','Last OSE']:np.testing.assert_array_equal(g.predicted,previous[previous.model.eq(name)].set_index('source_row').loc[g.source_row].predicted)
for subset,v in r['prediction'].items():
    keys=set(d.loc[d.eligible,'source_row']) if subset=='All eligible observations' else set(d.loc[d.change_stress_PSS.abs().ge(1),'source_row'])
    pp=p[p.source_row.isin(keys)]
    for name,g in pp.groupby('model'):
        np.testing.assert_allclose(v['scores'][name]['mae'],abs(g.predicted-g.observed).mean(),atol=1e-10)
    for label,c in v['comparisons'].items():
        a,b=label.split(' versus ');aa=pp[pp.model.eq(a)].set_index('source_row');bb=pp[pp.model.eq(b)].set_index('source_row').loc[aa.index]
        delta=abs(bb.observed-bb.predicted)-abs(aa.observed-aa.predicted);np.testing.assert_allclose(c['mae_improvement'],delta.mean(),atol=1e-10)
        gg=pd.DataFrame({'delta':delta,'group':aa.student_group}).groupby('group').delta.agg(['sum','count']);ix=np.random.default_rng(42).integers(len(gg),size=(3000,len(gg)));boot=gg['sum'].to_numpy()[ix].sum(axis=1)/gg['count'].to_numpy()[ix].sum(axis=1)
        np.testing.assert_allclose(c['person_cluster_95_interval'],np.quantile(boot,[.025,.975]),atol=1e-10)
path=study.ROOT/'outputs/ose-change-audit/paired_observations.csv';assert hashlib.sha256(path.read_bytes()).hexdigest()==r['descriptive']['source_sha256']
q=pd.read_csv(path,float_precision='round_trip').dropna(subset=['stress_PSS_before','stress_PSS_after']);x=q.stress_PSS_after-q.stress_PSS_before;y=q.OSE_after-q.OSE_before
np.testing.assert_allclose(r['descriptive']['correlation_stress_change_OSE_change'],np.corrcoef(x,y)[0,1],atol=1e-12)
for name,mask in [('Stress rose >=1',x.ge(1)),('Stress fell >=1',x.le(-1)),('Stress changed <1',x.abs().lt(1))]:
    v=r['descriptive']['groups'][name];assert v['pairs']==mask.sum() and v['ids']==q.loc[mask,'person'].nunique();np.testing.assert_allclose(v['mean_OSE_change'],y[mask].mean(),atol=1e-12);np.testing.assert_allclose(v['median_OSE_change'],y[mask].median(),atol=1e-12)
nozero=q.OSE_before.gt(0)&q.OSE_after.gt(0)
sensitivity={'excluding_zero_OSE_endpoints_pairs':int(nozero.sum()),'correlation_excluding_zero_OSE_endpoints':float(np.corrcoef(x[nozero],y[nozero])[0,1])}
(out/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
checks={'same_student_held_out_folds_and_no_student_overlap':True,'all_saved_predictions_reproduce':True,'imputation_fits_training_students_only':True,'no_current_OSE_or_OSE_change_predictors':True,'same1118_test_records_and_original_baselines':True,'all_MAE_differences_and_prediction_bootstrap_intervals_recomputed':True,'descriptive_change_correlation_and_group_means_recomputed':True,'zero_OSE_endpoint_sensitivity_saved':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps({'checks':checks,'sensitivity':sensitivity},indent=2))
