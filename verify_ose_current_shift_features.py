"""Independently rebuild all current snapshots and explicit non-OSE changes."""
import hashlib,json
import numpy as np
import pandas as pd
import ose_current_shift_models as study

OUT=study.OUT;d,schema,scopes,meta=study.load();source=study.features.SOURCE
for path,digest in meta['source_hashes'].items():assert hashlib.sha256(open(path,'rb').read()).hexdigest()==digest
old_schema,old_data,other=study.old.load();old_index=old_data.set_index('source_row');baseline=d.set_index('source_row')
np.testing.assert_array_equal(baseline.OSE,old_index.loc[baseline.index].OSE)
for col in study.old.previous.core.OSE_FEATURES:np.testing.assert_allclose(baseline[col],old_index.loc[baseline.index,col],equal_nan=True,atol=1e-10)
expected={};physical=pd.read_csv(source/'environment_daily_all.csv',float_precision='round_trip');physical.day=pd.to_datetime(physical.day);assert physical.slots.ge(72).all()
signals=['temperature_C','humidity','co2','voc','pm25','light','noise','device_score']
sensor_expected={c:[] for c in meta['physical_current_columns']}
for _,row in d.iterrows():
    g=physical[physical.person.eq(row.person)&physical.day.ge(row.date-pd.Timedelta(days=7))&physical.day.lt(row.date)]
    for signal in signals:
        if len(g)>=5:
            value=g[signal].to_numpy();days=(g.day-(row.date-pd.Timedelta(days=7))).dt.days.to_numpy(float);center=days-days.mean();vals=[value.mean(),np.sqrt(((value-value.mean())**2).mean()),np.dot(center,value-value.mean())/np.dot(center,center)*7]
        else:vals=[np.nan]*3
        for suffix,v in zip(['mean','sd','trend'],vals):sensor_expected['current_sensor_'+signal+'_'+suffix].append(v)
expected.update(sensor_expected)
specs=[('stress',study.ROOT/'outputs/expanded-stress-increases/cleaned_stress_reports.csv'),('mood',source/'cleaned_mood.csv'),('monthly',source/'cleaned_monthly.csv'),('computer',source/'cleaned_computer.csv'),('satisfaction',source/'cleaned_satisfaction.csv')]
for tag,path in specs:
    raw=pd.read_csv(path,float_precision='round_trip');raw.date=pd.to_datetime(raw.date)
    cols=[c for c in raw if c not in ['person','date','source_row','source_rows']]
    if tag=='satisfaction':cols=[c for c in cols if c not in meta['excluded_current_learning_impact_items']]
    lookup={}
    for col in cols:
        for person,g in raw[raw[col].notna()].groupby('person'):
            g=g.sort_values('date');lookup[(col,person)]=(g.date.to_numpy(),g[col].to_numpy())
    for col in cols:
        name='current_'+tag+'_'+study.old.previous.features.slug(col);values=[];ages=[]
        for _,row in d.iterrows():
            dates,v=lookup.get((col,row.person),(np.array([],dtype='datetime64[ns]'),np.array([])));n=np.searchsorted(dates,np.datetime64(row.date,'ns'),side='right')
            if n:values.append(v[n-1]);ages.append((row.date-pd.Timestamp(dates[n-1])).days);assert pd.Timestamp(dates[n-1])<=row.date
            else:values.append(np.nan);ages.append(np.nan)
        expected[name]=values;expected[name+'_age_days']=ages
raw=pd.read_csv(source/'fitbit_daily_all.csv',float_precision='round_trip');raw.day=pd.to_datetime(raw.day);cols=[c for c in raw if c.startswith('fit_')];wearable={c:[] for col in cols for c in ['current_'+col+'_week_mean','current_'+col+'_between_day_sd']}
for _,row in d.iterrows():
    g=raw[raw.person.eq(row.person)&raw.day.ge(row.date-pd.Timedelta(days=8))&raw.day.lt(row.date-pd.Timedelta(days=1))]
    for col in cols:
        values=g[col].dropna().to_numpy();mu=values.mean() if len(values)>=3 else np.nan
        wearable['current_'+col+'_week_mean'].append(mu);wearable['current_'+col+'_between_day_sd'].append(np.sqrt(np.mean((values-mu)**2)) if len(values)>=3 else np.nan)
expected.update(wearable)
raw=pd.read_csv(source/'cleaned_diary.csv',float_precision='round_trip');raw.date=pd.to_datetime(raw.date);raw['event_day']=raw.date-pd.Timedelta(days=1);cols=[c for c in raw if c not in ['person','date','source_rows','event_day']]
diary={f'current_diary_item{i}_code{code}_fraction':[] for i in range(1,len(cols)+1) for code in range(5)}
for _,row in d.iterrows():
    g=raw[raw.person.eq(row.person)&raw.event_day.ge(row.date-pd.Timedelta(days=8))&raw.event_day.lt(row.date-pd.Timedelta(days=1))&raw.date.lt(row.date)]
    for i,col in enumerate(cols,1):
        values=g[col];values=values[values.isin(range(5))].to_numpy()
        for code in range(5):diary[f'current_diary_item{i}_code{code}_fraction'].append(np.count_nonzero(values==code)/len(values) if len(values)>=3 else np.nan)
expected.update(diary)
raw=pd.read_csv(source/'facial_daily.csv',float_precision='round_trip');raw.day=pd.to_datetime(raw.day);cols=[c for c in raw if c.startswith('face_')];rows=[]
for _,row in d.iterrows():
    g=raw[raw.person.eq(row.person)&raw.day.ge(row.date-pd.Timedelta(days=28))&raw.day.lt(row.date)].sort_values('day');rows.append(g.iloc[-1][cols].to_numpy(float) if len(g) else np.full(len(cols),np.nan))
face=pd.DataFrame(rows,columns=['current_'+c for c in cols]);current=pd.concat([pd.DataFrame(expected),face],axis=1)
for col in current:np.testing.assert_allclose(d[col],current[col],equal_nan=True,rtol=1e-9,atol=1e-9)
prev=current.groupby(d.person).shift(1);gap=d.groupby('person').date.diff().dt.days;valid=gap.between(7,21)
for col in meta['delta_source_columns']:
    contrast=(current[col]-prev[col]).where(valid);name='change_'+col.removeprefix('current_');np.testing.assert_allclose(d[name],contrast,equal_nan=True,atol=1e-9,rtol=1e-9)
for columns in scopes.values():assert not set(columns)&{'OSE','OSE_before','OSE_after','OSE_change','person','source_row','student_group','eligible','date'}
assert not any('current_satisfaction_'+study.old.previous.features.slug(c) in meta['new_columns'] for c in meta['excluded_current_learning_impact_items'])
audit=pd.read_csv(OUT/'source_timing_audit.csv');valid=audit.latest_date.notna();dates=pd.to_datetime(audit.loc[valid,'latest_date']);bounds=pd.to_datetime(audit.loc[valid,'bound']);inclusive=audit.loc[valid,'bound_inclusive'].to_numpy(bool)
assert np.where(inclusive,(dates<=bounds).to_numpy(),(dates<bounds).to_numpy()).all()
checks={'all1530_current_sensor_snapshots_recomputed':True,'all_current_non_OSE_survey_values_and_ages_rebuilt_from_sources':True,'all_current_Fitbit_and_diary_summaries_rebuilt':True,'all_current_facial_selection_rebuilt':True,'all_explicit_non_OSE_changes_recomputed_with7to21day_gaps':True,'source_dates_obey_documented_bounds':True,'original_OSE_history_and_target_values_unchanged':True,'no_target_OSE_or_OSE_change_in_predictors':True,'current_learning_impact_questionnaire_items_excluded':True,'all_input_source_hashes_unchanged':True}
(OUT/'feature_independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
