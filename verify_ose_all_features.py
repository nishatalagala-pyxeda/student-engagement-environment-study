"""Independent recomputation of all dated non-outcome predictor summaries."""
import json
import numpy as np
import pandas as pd
import ose_all_features as f
OUT=f.OUT;schema=json.loads((OUT/'feature_schema.json').read_text());d=pd.read_csv(OUT/'modeling_features.csv',float_precision='round_trip',dtype={c:str for c in schema['categorical_features']});d.date=pd.to_datetime(d.date)

def field_stores():
    result={}
    for tag,meta in schema['source_metadata'].items():
        if tag in ['demographics','diary']:continue
        raw=pd.read_csv(meta['path']);datecol=next(c for c in raw if 'date' in c.lower());raw['person']=raw.ID.map(f.core.canonical);raw['date']=pd.to_datetime(raw[datecol],format='mixed',errors='coerce').dt.normalize();raw=raw[raw.person.notna()&raw.date.notna()]
        for c in meta['columns']:raw[c]=pd.to_numeric(raw[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        collected={}
        for (person,date),g in raw.groupby(['person','date'],sort=True):
            for c in meta['columns']:
                values=np.unique(g[c].dropna().to_numpy())
                if len(values)==1:collected.setdefault((person,c),[]).append((np.datetime64(date,'ns'),float(values[0])))
        for (person,c),pairs in collected.items():result[(tag,person,c)]=(np.array([v[0] for v in pairs]),np.array([v[1] for v in pairs]))
    return result

def check_episodic(frame,stores,frozen=False):
    cols=schema['episodic_features'];expected=[]
    for _,r in frame.iterrows():
        bound=pd.Timestamp(r.date)-pd.Timedelta(days=7)
        if frozen:bound=min(bound,pd.Timestamp(r.cutoff_exclusive))
        values={}
        for tag,meta in schema['source_metadata'].items():
            if tag in ['demographics','diary']:continue
            for c in meta['columns']:
                dates,y=stores.get((tag,r.person,c),(np.array([],dtype='datetime64[ns]'),np.array([])));n=np.searchsorted(dates,np.datetime64(bound,'ns'),side='left');v=y[:n];key=tag+'_'+f.slug(c)
                if n:mu=sum(v)/n;out=[v[-1],mu,np.sqrt(sum((v-mu)**2)/n),(pd.Timestamp(r.date)-pd.Timestamp(dates[n-1])).days]
                else:out=[np.nan]*4
                values.update({key+'_'+s:x for s,x in zip(['last','mean','sd','age_days'],out)})
        expected.append([values[c] for c in cols])
    np.testing.assert_allclose(frame[cols].to_numpy(float),np.array(expected),equal_nan=True,atol=1e-9,rtol=1e-9)

stores=field_stores();check_episodic(d,stores);print('Full episodic sources and temporal summaries checked',flush=True)
# Diary: independently rebuild per-date nonconflicting codes from the raw source.
raw=pd.read_csv(schema['source_metadata']['diary']['path']);raw['person']=raw.ID.map(f.core.canonical);raw['date']=pd.to_datetime(raw['Start Date'],format='mixed',errors='coerce');items=schema['source_metadata']['diary']['columns'];diaries={}
for (person,date),g in raw[raw.person.notna()&raw.date.notna()].groupby(['person','date']):
    for i,c in enumerate(items,1):
        v=np.unique(pd.to_numeric(g[c],errors='coerce').dropna())
        if len(v)==1 and v[0] in range(5):diaries.setdefault((person,i),[]).append((date,float(v[0])))
for _,r in d.iterrows():
    start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8)
    for i in range(1,len(items)+1):
        vals=[v for date,v in diaries.get((r.person,i),[]) if start<=date-pd.Timedelta(days=1)<end and date<r.date-pd.Timedelta(days=7)]
        for code in range(5):np.testing.assert_allclose(r[f'diary_item{i}_code{code}_fraction'],np.mean(np.array(vals)==code) if len(vals)>=3 else np.nan,equal_nan=True,atol=1e-12)
print('Diary raw codes, duplicates and windows checked',flush=True)
# Recompute every dated sensor, personal-reference, wearable and facial selection.
env=pd.read_csv(OUT/'environment_daily_all.csv',float_precision='round_trip');env.day=pd.to_datetime(env.day);eb={p:g for p,g in env.groupby('person')}
fit=pd.read_csv(OUT/'fitbit_daily_all.csv',float_precision='round_trip');fit.day=pd.to_datetime(fit.day);fb={p:g for p,g in fit.groupby('person')};fc=[c for c in fit if c.startswith('fit_')]
face=pd.read_csv(OUT/'facial_daily.csv',float_precision='round_trip');face.day=pd.to_datetime(face.day);faceb={p:g.sort_values('day') for p,g in face.groupby('person')};facecols=schema['face_features'];signals=['temperature_C','humidity','co2','voc','pm25','light','noise','device_score']
for _,r in d.iterrows():
    start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8);g=eb.get(r.person,env.iloc[:0]);recent=g[g.day.ge(start)&g.day.lt(end)];ref=g[g.day.ge(start-pd.Timedelta(days=28))&g.day.lt(start)]
    for s in signals:
        mu=recent[s].mean() if len(recent)>=5 else np.nan;sd=recent[s].std(ddof=0) if len(recent)>=5 else np.nan
        values=[mu,sd,mu-ref[s].mean() if len(recent)>=5 and len(ref)>=14 else np.nan,sd-ref[s].std(ddof=0) if len(recent)>=5 and len(ref)>=14 else np.nan,np.polyfit((recent.day-start).dt.days,recent[s],1)[0]*7 if len(recent)>=5 else np.nan]
        for suffix,v in zip(['mean','sd','mean_departure','sd_departure','trend'],values):np.testing.assert_allclose(r['env_'+s+'_'+suffix],v,equal_nan=True,atol=1e-9,rtol=1e-9)
    g=fb.get(r.person,fit.iloc[:0]);g=g[g.day.ge(start)&g.day.lt(end)]
    for c in fc:
        v=g[c].dropna().to_numpy();expected=[sum(v)/len(v),np.std(v,ddof=0)] if len(v)>=3 else [np.nan,np.nan]
        for suffix,value in zip(['week_mean','between_day_sd'],expected):np.testing.assert_allclose(r[c+'_'+suffix],value,equal_nan=True,atol=1e-8,rtol=1e-9)
    g=faceb.get(r.person,face.iloc[:0]);g=g[g.day.ge(end-pd.Timedelta(days=28))&g.day.lt(end)]
    want=g.iloc[-1][facecols].to_numpy(float) if len(g) else np.full(len(facecols),np.nan);np.testing.assert_allclose(r[facecols].to_numpy(float),want,equal_nan=True,atol=1e-10)
print('All daily modality windows and summaries checked',flush=True)
# Verify demographic values against the original workbook; no identifiers become predictors.
demo=pd.read_excel(schema['source_metadata']['demographics']['path']);demo['person']=demo.ID.map(f.core.canonical);demo=demo.set_index('person')
for _,r in d.iterrows():
    if r.person not in demo.index:continue
    v=demo.loc[r.person]
    for c in demo:
        name='demo_'+f.slug(c)
        if c=='ID' or name not in d:continue
        if c in ['YearBirth','Ladder']:
            n=pd.to_numeric(pd.Series([v[c]]),errors='coerce').iloc[0];n=r.date.year-n if c=='YearBirth' else n;np.testing.assert_allclose(r[name],n,equal_nan=True,atol=1e-12)
        elif pd.notna(v[c]):assert str(r[name])==str(v[c]).strip()
for name,cols in schema['groups'].items():assert not set(cols)&{'OSE','person','ID','source_row','date'}
checks={'all1530_full_episodic_features_recomputed_from_original_sources':True,'all1530_diary_features_rebuilt_from_raw_codes':True,'all1530_environment_and_personal_reference_summaries_recomputed':True,'all1530_Fitbit_weekly_summaries_recomputed':True,'all1530_facial_session_selections_recomputed':True,'demographic_matches_checked_against_original_workbook':True,'identifiers_and_target_excluded_from_predictor_groups':True}
(OUT/'feature_independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
