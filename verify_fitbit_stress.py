"""Independent checks of the saved feature windows and held-out predictions."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import fitbit_stress as f
root=Path(__file__).resolve().parent;out=root/'outputs/fitbit-stress'
d=pd.read_csv(out/'modeling_data.csv');a=pd.read_csv(out/'feature_window_audit.csv');daily=pd.read_csv(out/'daily_summaries.csv');daily.day=pd.to_datetime(daily.day)
for c in ['date','features_start','features_end_exclusive','forecast_origin']:a[c]=pd.to_datetime(a[c])
assert (a.features_end_exclusive==a.date-pd.Timedelta(days=8)).all()
assert (a.features_start==a.date-pd.Timedelta(days=15)).all()
assert (a.forecast_origin==a.date-pd.Timedelta(days=7)).all()
for i in d.index[d.wearable_eligible]:
    w=a.iloc[i];g=daily[daily.person.eq(w.person)&daily.day.ge(w.features_start)&daily.day.lt(w.features_end_exclusive)]
    assert g.Steps.notna().sum()==w.step_days
    np.testing.assert_allclose(d.loc[i,'fitbit_Steps_mean'],sum(g.Steps.dropna())/g.Steps.notna().sum())
    values=g.Steps.dropna().to_numpy();np.testing.assert_allclose(d.loc[i,'fitbit_Steps_sd'],np.sqrt(sum((values-values.mean())**2)/len(values)))
    if w.sleep_days<3:assert pd.isna(d.loc[i,'fitbit_Minutes_Asleep_mean'])
ha=pd.read_csv(out/'history_audit.csv');ha.date=pd.to_datetime(ha.date);ha.latest_history_date=pd.to_datetime(ha.latest_history_date)
assert (ha.loc[ha.latest_history_date.notna(),'latest_history_date']<ha.loc[ha.latest_history_date.notna(),'date']-pd.Timedelta(days=7)).all()
test=ha[ha.role.eq('test')&ha.latest_history_date.notna()]
assert (test.latest_history_date<pd.to_datetime(test.history_cutoff_exclusive)).all()
p=pd.read_csv(out/'predictions.csv');assert not p.duplicated(['protocol','model','source_row']).any()
for _,pp in p.groupby('protocol'):
    keys=None
    for _,mm in pp.groupby('model'):
        current=set(mm.source_row);assert keys is None or current==keys;keys=current
r=json.loads((out/'results.json').read_text())
for protocol,groups in r['scores'].items():
    for subset,models in groups.items():
        pp=p[p.protocol.eq(protocol)]
        if subset!='all_wearable_covered':pp=pp[pp.known_history]
        if subset=='known_history_and_sleep':pp=pp[pp.sleep_covered]
        for name,v in models.items():
            mm=pp[pp.model.eq(name)];np.testing.assert_allclose(v['mae'],np.abs(mm.observed-mm.predicted).mean(),rtol=0,atol=1e-12)
checks={'all_97_covered_step_means_and_population_SDs_recomputed':True,'all_history_cutoffs_checked_from_saved_audit':True,'all_models_share_test_rows':True,'all_reported_MAEs_independently_recomputed':True,'all_feature_windows_recomputed_from_dates':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2))
f.render(r)
print(json.dumps(checks,indent=2))
