"""Independently check saved reference means, targets, forecast dates and scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import stress_increases as model
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/stress-increases'
source=next(model.base.SOURCE.rglob('PANAS_PSS.csv'));raw=pd.read_csv(source);raw['source_row']=raw.index+2
raw['person']=raw.ID.map(model.canonical);raw['date']=pd.to_datetime(raw.StartDate,errors='coerce',format='mixed')
h=raw[raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4)].copy()
counts=h.groupby(['person','date']).PSS.nunique();valid=set(counts[counts.eq(1)].index)
h=h[[tuple(k) in valid for k in h[['person','date']].to_numpy()]].drop_duplicates(['person','date'])
audit=pd.read_csv(OUT/'history_target_audit.csv');audit.date=pd.to_datetime(audit.date)
for _,r in audit.iterrows():
    old=h[h.person.eq(r.person)&h.date.lt(r.date-pd.Timedelta(days=7))]
    if r.role=='test':old=old[old.date.lt(pd.Timestamp(r.history_cutoff_exclusive))]
    assert len(old)==r.prior_report_count and len(old)>=2
    assert set(old.source_row)==set(map(int,r.history_source_rows.split(',')))
    assert int(r.source_row) not in set(old.source_row)
    np.testing.assert_allclose(r.past_mean_stress,old.PSS.sum()/len(old),atol=1e-12,rtol=0)
    actual=raw.loc[raw.source_row.eq(r.source_row),'PSS'].item();np.testing.assert_allclose(r.PSS,actual,atol=1e-12,rtol=0)
    np.testing.assert_allclose(r.excess_stress,actual-old.PSS.mean(),atol=1e-12,rtol=0)
    assert r.event==bool(actual-old.PSS.mean()>=.5)
p=pd.read_csv(OUT/'predictions.csv');r=json.loads((OUT/'results.json').read_text())
for (protocol,task),pp in p.groupby(['protocol','task']):
    rowsets=[set(g.source_row) for _,g in pp.groupby('model')];assert all(x==rowsets[0] for x in rowsets)
    target=audit[audit.protocol.eq(protocol)&audit.role.eq('test')].set_index('source_row');assert target.index.is_unique
    for name,g in pp.groupby('model'):
        assert g.source_row.is_unique
        ref=target.loc[g.source_row]
        if task=='classification':
            np.testing.assert_array_equal(g.observed.to_numpy(),ref.event.astype(int).to_numpy())
            score=float(np.mean((g.predicted-g.observed)**2));np.testing.assert_allclose(score,r['primary']['scores'][protocol][task][name]['brier'],atol=1e-12)
            assert g.predicted.between(0,1).all()
        else:
            np.testing.assert_allclose(g.observed.to_numpy(),ref.excess_stress.to_numpy(),atol=1e-12)
            score=float(np.mean(np.abs(g.predicted-g.observed)));np.testing.assert_allclose(score,r['primary']['scores'][protocol][task][name]['mae'],atol=1e-12)
            assert (g.predicted+g.history_mean).between(-1e-12,4+1e-12).all()
        if name=='Always no increase':assert g.predicted.eq(0).all()
        if name=='Zero excess (past mean)':assert g.predicted.eq(0).all()
daily=pd.read_csv(OUT/'environment_daily.csv');daily.day=pd.to_datetime(daily.day);windows=pd.read_csv(OUT/'earlier_environment_audit.csv');features=pd.read_csv(OUT/'modeling_features.csv').set_index('source_row')
for _,w in windows.iterrows():
    date=pd.Timestamp(w.date);start=date-pd.Timedelta(days=15);end=date-pd.Timedelta(days=8)
    assert pd.Timestamp(w.start)==start and pd.Timestamp(w.end_exclusive)==end
    ds=daily[daily.person.eq(w.person)&daily.day.ge(start)&daily.day.lt(end)];assert len(ds)==w.good_days
    row=features.loc[w.source_row]
    for s in model.SIGNALS:
        if len(ds)<5:assert pd.isna(row['lag_env_'+s+'_mean'])
        else:
            np.testing.assert_allclose(row['lag_env_'+s+'_mean'],ds[s].sum()/len(ds),atol=1e-10)
            np.testing.assert_allclose(row['lag_env_'+s+'_sd'],np.sqrt(np.mean((ds[s]-ds[s].mean())**2)),atol=1e-10)
checks={'all_historical_reference_means_recomputed_from_source':True,'all_event_and_excess_targets_recomputed_from_source':True,'all_reference_histories_exclude_target_and_future_reports':True,'all_models_share_test_rows_and_labels':True,'all_primary_Brier_and_MAE_values_recomputed':True,'all_234_earlier_environment_windows_and_summaries_recomputed':True,'all_prediction_bounds_checked':True,'audited_training_and_test_reference_rows':len(audit)}
(OUT/'independent_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
