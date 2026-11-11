"""Source-window and paired-score checks for environmental additions."""
import json
import numpy as np
import pandas as pd
import expanded_stress_increases as core
OUT=core.OUT
d=pd.read_csv(OUT/'environment_modeling_features.csv');daily=pd.read_csv(core.ROOT/'outputs/stress-increases/environment_daily.csv');daily.day=pd.to_datetime(daily.day)
for _,row in d.iterrows():
    date=pd.Timestamp(row.date);ds=daily[daily.person.eq(row.person)&daily.day.ge(date-pd.Timedelta(days=15))&daily.day.lt(date-pd.Timedelta(days=8))]
    assert row.environment_covered==bool(len(ds)>=5)
    for name in core.old.SIGNALS:
        if len(ds)<5:assert pd.isna(row['lag_env_'+name+'_mean']) and pd.isna(row['lag_env_'+name+'_sd'])
        else:
            np.testing.assert_allclose(row['lag_env_'+name+'_mean'],ds[name].sum()/len(ds),atol=1e-10)
            np.testing.assert_allclose(row['lag_env_'+name+'_sd'],np.sqrt(np.mean((ds[name]-ds[name].mean())**2)),atol=1e-10)
p=pd.read_csv(OUT/'environment_comparison_predictions.csv',float_precision='round_trip');r=json.loads((OUT/'environment_results.json').read_text())
for protocol,pp in p.groupby('protocol'):
    for subset,sub in [('measured_earlier_environment',pp[pp.environment_covered]),('all_test_reports',pp)]:
        for task,taskp in sub.groupby('task'):
            sets=[set(g.source_row) for _,g in taskp.groupby('model')];assert all(x==sets[0] for x in sets)
            for name,mp in taskp.groupby('model'):
                assert mp.source_row.is_unique;v=r['scores'][protocol][subset][task][name]
                if task=='classification':
                    np.testing.assert_allclose(v['brier'],np.mean((mp.observed-mp.predicted)**2),atol=1e-12);np.testing.assert_array_equal(mp.observed.to_numpy(),(mp.observed_PSS-mp.history_mean).ge(.5).to_numpy(float))
                else:
                    np.testing.assert_allclose(v['mae'],np.mean(abs(mp.observed-mp.predicted)),atol=1e-12);assert (mp.predicted+mp.history_mean).between(-1e-12,4+1e-12).all()
            for family in ['linear','forest']:
                a=taskp[taskp.model.eq('History + environment | '+family)].set_index('source_row');b=taskp[taskp.model.eq('History | '+family)].set_index('source_row').loc[a.index]
                np.testing.assert_allclose(a.observed,b.observed,atol=1e-12);np.testing.assert_allclose(a.history_mean,b.history_mean,atol=1e-12)
checks={'all1534_prior_environment_windows_and_coverage_checked':True,'all_available_environment_mean_SD_features_recomputed':True,'test_row_target_reference_equality_across_nested_models':True,'all_environment_comparison_Brier_and_MAE_recomputed':True,'regression_bounds_checked':True}
(OUT/'environment_independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
