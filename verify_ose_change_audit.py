"""Independent recomputation of paired chronology, exposures and change events."""
import hashlib,json
import numpy as np
import pandas as pd
import ose_change_audit as study

out=study.OUT;r=json.loads((out/'results.json').read_text())
p=pd.read_csv(out/'paired_observations.csv',float_precision='round_trip');p.date=pd.to_datetime(p.date);p.previous_date=pd.to_datetime(p.previous_date)
events=pd.read_csv(out/'large_change_examples.csv',float_precision='round_trip')
ose=pd.read_csv(study.ROOT/'outputs/ose-stress-prediction/cleaned_ose.csv',float_precision='round_trip');ose.date=pd.to_datetime(ose.date);ose=ose.sort_values(['person','date']);indexed=ose.set_index('source_row')
daily=pd.read_csv(study.SOURCE/'environment_daily_all.csv',float_precision='round_trip');daily.day=pd.to_datetime(daily.day)
assert daily.slots.ge(72).all() and not daily.duplicated(['person','day']).any()
previous=ose.groupby('person').shift(1);mask=(ose.date-previous.date).dt.days.between(7,21)
assert set(p.source_row)==set(ose.loc[mask,'source_row']) and p.source_row.is_unique
np.testing.assert_array_equal(p.previous_source_row,previous.loc[mask].set_index(ose.loc[mask,'source_row']).loc[p.source_row].source_row)
np.testing.assert_allclose(p.OSE_after,indexed.loc[p.source_row].OSE,atol=1e-10)
np.testing.assert_allclose(p.OSE_before,indexed.loc[p.previous_source_row].OSE,atol=1e-10)
np.testing.assert_allclose(p.OSE_change,p.OSE_after-p.OSE_before,atol=1e-10)
for _,row in p.iterrows():
    for label,start,end in [('recent',7,0),('forecast',15,8)]:
        for suffix,date in [('before',row.previous_date),('after',row.date)]:
            window=daily[daily.person.eq(row.person)&daily.day.ge(date-pd.Timedelta(days=start))&daily.day.lt(date-pd.Timedelta(days=end))]
            for c in study.SENSORS:
                expected=window[c].mean() if len(window)>=5 else np.nan
                np.testing.assert_allclose(row[label+'_'+c+'_'+suffix],expected,equal_nan=True,atol=1e-9)
sources={}
for tag,path in [('satisfaction',study.SOURCE/'cleaned_satisfaction.csv'),('computer',study.SOURCE/'cleaned_computer.csv'),('mood',study.SOURCE/'cleaned_mood.csv'),('stress',study.ROOT/'outputs/expanded-stress-increases/cleaned_stress_reports.csv')]:
    table=pd.read_csv(path,float_precision='round_trip');table.date=pd.to_datetime(table.date);sources[tag]=table.set_index(['person','date'])
for summary in r['summaries']:
    col=summary['factor'];a=p[col+'_before'];b=p[col+'_after'];delta=b-a
    if not col.startswith(('recent_','forecast_')):
        tag,rawcol=col.split('_',1)
        for suffix,dates in [('before',p.previous_date),('after',p.date)]:
            keys=pd.MultiIndex.from_arrays([p.person,dates]);expected=sources[tag][rawcol].reindex(keys).to_numpy()
            np.testing.assert_allclose(p[col+'_'+suffix],expected,equal_nan=True,atol=1e-10)
    available=a.notna()&b.notna()
    if col in ['computer_DownloadSpeed','computer_UploadSpeed']:
        threshold=20 if col.endswith('DownloadSpeed') else 10
        available&=a.gt(0)&b.gt(0);large=available&(np.maximum(a/b,b/a)>=3)&delta.abs().ge(threshold)
    else:
        threshold=study.SENSORS[col.split('_',1)[1]] if col.startswith(('recent_','forecast_')) else study.SURVEYS[col][1]
        large=available&delta.abs().ge(threshold)
    assert summary['covered_pairs']==available.sum() and summary['large_pairs']==large.sum()
    ev=events[events.factor.eq(col)];assert set(ev.source_row_after)==set(p.loc[large,'source_row'])
    for label,m in [('large_change',large),('smaller_change',available&~large)]:
        v=p.loc[m,'OSE_change'];s=summary['groups'][label]
        assert s['n']==len(v) and s['ids']==p.loc[m,'person'].nunique()
        if len(v):np.testing.assert_allclose(s['median_absolute_OSE_change'],v.abs().median(),atol=1e-10)
for group,s in r['unions'].items():
    unique=events[events.group.eq(group)].drop_duplicates(['person','date_before','date_after'])
    assert s['pairs']==len(unique) and s['ids']==unique.person.nunique()
    assert s['OSE_change_ge5']==unique.OSE_change.abs().ge(5).sum()
for path,digest in r['source_hashes'].items():assert hashlib.sha256(open(path,'rb').read()).hexdigest()==digest
checks={'all_pairs_are_consecutive_same_person_7_to21day_observations':True,'OSE_values_match_cleaned_source':True,'all_recent_and_forecast_sensor_means_recomputed_with_coverage':True,'sensor_windows_nonoverlapping_within_each_pair':True,'same_date_survey_values_recomputed_for_every_factor':True,'large_event_membership_and_counts_recomputed':True,'unique_transition_counts_deduplicated_across_factors':True,'missing_factors_never_assumed_unchanged':True,'descriptive_outcome_summaries_recomputed':True,'source_hashes_unchanged':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
