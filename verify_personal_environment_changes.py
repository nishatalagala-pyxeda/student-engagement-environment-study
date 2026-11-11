"""Independently recompute personal references, departures, trends and metrics."""
import json
import numpy as np
import pandas as pd
import personal_environment_changes as analysis
OUT=analysis.OUT
d=pd.read_csv(OUT/'modeling_features.csv',float_precision='round_trip');a=pd.read_csv(OUT/'personal_reference_audit.csv');daily=pd.read_csv(analysis.ROOT/'outputs/stress-increases/environment_daily.csv',float_precision='round_trip');daily.day=pd.to_datetime(daily.day)
assert d.source_row.is_unique and a.source_row.is_unique;aud=a.set_index('source_row');count=0
for _,row in d.iterrows():
    date=pd.Timestamp(row.date);start=date-pd.Timedelta(days=15);end=date-pd.Timedelta(days=8);reference_start=date-pd.Timedelta(days=43)
    person=daily[daily.person.eq(row.person)];recent=person[person.day.ge(start)&person.day.lt(end)];reference=person[person.day.ge(reference_start)&person.day.lt(start)]
    audit=aud.loc[row.source_row];eligible=len(recent)>=5 and len(reference)>=14
    assert bool(row.personal_environment_eligible)==eligible==bool(audit.personal_environment_eligible)
    assert len(recent)==audit.recent_good_days and len(reference)==audit.reference_good_days
    assert pd.Timestamp(audit.reference_end_exclusive)==start and pd.Timestamp(audit.recent_end_exclusive)==end
    if eligible:
        assert reference.day.max()<recent.day.min() and recent.day.max()<date-pd.Timedelta(days=7)
        count+=1
    for signal in analysis.SIGNALS:
        if not eligible:
            assert all(pd.isna(row['change_'+signal+'_'+suffix]) for suffix in ['mean_departure','sd_departure','trend_per_week']);continue
        mu=sum(reference[signal])/len(reference);sd=np.sqrt(sum((reference[signal]-mu)**2)/len(reference))
        np.testing.assert_allclose(audit['reference_'+signal+'_mean'],mu,atol=1e-10);np.testing.assert_allclose(audit['reference_'+signal+'_sd'],sd,atol=1e-10)
        np.testing.assert_allclose(row['change_'+signal+'_mean_departure'],sum(recent[signal])/len(recent)-mu,atol=1e-10)
        np.testing.assert_allclose(row['change_'+signal+'_sd_departure'],np.std(recent[signal].to_numpy(),ddof=0)-sd,atol=1e-10)
        # Independent polynomial least squares checks centered-covariance slopes.
        x=(recent.day-start).dt.days.to_numpy(float);slope=np.polyfit(x,recent[signal].to_numpy(float),1)[0]
        np.testing.assert_allclose(row['change_'+signal+'_trend_per_week'],slope*7,atol=1e-9)
p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');r=json.loads((OUT/'results.json').read_text());earlier=pd.read_csv(analysis.core.OUT/'environment_comparison_predictions.csv',float_precision='round_trip')
for (protocol,task,name),g in p.groupby(['protocol','task','model']):
    assert g.source_row.is_unique
    prior=earlier[earlier.protocol.eq(protocol)&earlier.task.eq(task)]
    if 'changes' not in name:
        prior=prior[prior.model.eq(name)].set_index('source_row').loc[g.source_row];np.testing.assert_array_equal(g.predicted.to_numpy(),prior.predicted.to_numpy())
    prior=earlier[earlier.protocol.eq(protocol)&earlier.task.eq(task)&earlier.model.eq('History | linear')].set_index('source_row').loc[g.source_row]
    np.testing.assert_allclose(g.observed.to_numpy(),prior.observed.to_numpy(),atol=1e-12);np.testing.assert_allclose(g.history_mean.to_numpy(),prior.history_mean.to_numpy(),atol=1e-12)
for protocol,pp in p.groupby('protocol'):
    for subset,ss in [('personal_environment_measured',pp[pp.personal_environment_eligible]),('all_test_reports',pp)]:
        for task,taskp in ss.groupby('task'):
            sets=[set(g.source_row) for _,g in taskp.groupby('model')];assert all(s==sets[0] for s in sets)
            for name,g in taskp.groupby('model'):
                v=r['scores'][protocol][subset][task][name]
                if task=='classification':
                    np.testing.assert_allclose(v['brier'],np.mean((g.observed-g.predicted)**2),atol=1e-12);assert g.predicted.between(0,1).all()
                else:
                    np.testing.assert_allclose(v['mae'],np.mean(abs(g.observed-g.predicted)),atol=1e-12);assert (g.predicted+g.history_mean).between(-1e-12,4+1e-12).all()
checks={'all1534_reference_recent_windows_and_coverage_recomputed':True,'all_personal_baseline_means_SDs_and_departures_recomputed':True,'all_eligible_trends_recomputed_by_independent_polynomial_fit':True,'no_overlap_of_reference_recent_or_target_recall_windows':True,'earlier_saved_model_predictions_reproduce_exactly':True,'same_test_rows_labels_and_historical_stress_references':True,'all_Brier_and_MAE_values_recomputed':True,'prediction_bounds_checked':True,'personal_reference_eligible_reports':count}
(OUT/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
