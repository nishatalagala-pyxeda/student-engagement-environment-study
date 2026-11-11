"""Current non-OSE observations and explicit changes, with source timing audit."""
import hashlib,json
import numpy as np
import pandas as pd
import ose_unseen_students as old

ROOT=old.ROOT;SOURCE=old.SOURCE;OUT=ROOT/'outputs/ose-current-shifts'
PLAN={
 'question':'Does supplying current non-OSE conditions and explicit changes improve held-out-student OSE estimates, especially on substantial-shift observations?',
 'interpretation':'Concurrent assisted OSE estimation, not a week-ahead forecast. Non-OSE surveys dated on/before target can be supplied; same-day ordering is unknown. Target OSE, its item responses, current OSE change and future observations never predictors.',
 'cohort':'Original1118 history-eligible reports138 cohort-aware student IDs. Exact previous five student-held-out folds; conservative same numeric suffix across cohorts kept together. OSE history unchanged: same-person reports strictly before D-7.',
 'physical':'Eight available sensor measurements. Equal-weight daily mean, between-day SD and per7day linear trend over[D-7,D); >=5good days, each already>=72slots. Survey day excluded. Sensor ID links inferred, occupancy unknown, noise arithmetic notLeq. No new archive extraction.',
 'surveys':'Latest valid observation on/before D per non-OSE questionnaire variable, plus age: PSS, PANAS, monthly CESD/STAI raw codes, computer/network, environmental satisfaction. Exclude current environmental Enh/Int and Incr/Decr items because they ask whether conditions help learning. Unknown questionnaire aggregation/coding limits remain.',
 'daily':'Fitbit available daily summaries and diary code fractions over[D-8,D-1), retaining one complete day buffer. Each summary needs>=3valid days. Facial latest valid daily summary in[D-28,D), no same-day recording; same training-only PCA10. No physiological/emotion-label interpretation.',
 'changes':'For each current non-facial value/summary, subtract its snapshot at the immediately previous OSE report date, only if date gap7–21days. Missing either endpoint=>missing change. No subtraction of target OSE; no OSE-change input. Facial old/current blocks enter PCA, no raw geometry delta block.',
 'subgroups':'Reuse the prior audit large-shift memberships: physical current-week, current environmental-satisfaction, other current survey/network, and their union. Shift selection uses only factor contrasts, never OSE movement. Thresholds unchanged. Same cases across compared models; subgroups overlap.',
 'limits':'Repeated exploratory dataset, not external/future-semester validation. Same-day psychological answers can share response bias with engagement. Sparse physical sensor cohort and inferred links. New measurements must actually be collected at deployment. Selection and source-quality limits remain.'
}

def build():
    OUT.mkdir(parents=True,exist_ok=True);schema,d,other=old.load();d=d.sort_values(['person','date']).reset_index(drop=True)
    files=[SOURCE/'modeling_features.csv',SOURCE/'feature_schema.json'];current={};audits=[];physical=[];environment=[]
    daily=pd.read_csv(SOURCE/'environment_daily_all.csv',float_precision='round_trip');daily.day=pd.to_datetime(daily.day);files.append(SOURCE/'environment_daily_all.csv')
    signals=['temperature_C','humidity','co2','voc','pm25','light','noise','device_score'];byid={p:g for p,g in daily.groupby('person')};rows=[]
    for _,r in d.iterrows():
        g=byid.get(r.person,pd.DataFrame(columns=daily.columns));g=g[g.day.ge(r.date-pd.Timedelta(days=7))&g.day.lt(r.date)];rec={}
        for signal in signals:
            names=['current_sensor_'+signal+'_'+s for s in ['mean','sd','trend']]
            if len(g)>=5:
                x=(g.day-(r.date-pd.Timedelta(days=7))).dt.days.to_numpy(float);xc=x-x.mean();v=g[signal].to_numpy(float)
                values=[np.mean(v),np.std(v),float(xc@(v-v.mean())/(xc@xc)*7)]
            else:values=[np.nan]*3
            rec.update(dict(zip(names,values)))
        rows.append(rec);audits.append({'source_row':int(r.source_row),'source':'sensors','count':len(g),'latest_date':str(g.day.max().date()) if len(g) else None,'bound':str(r.date.date()),'bound_inclusive':False})
    for col in rows[0]:current[col]=[row[col] for row in rows];physical.append(col)
    specs=[('stress',ROOT/'outputs/expanded-stress-increases/cleaned_stress_reports.csv'),('mood',SOURCE/'cleaned_mood.csv'),('monthly',SOURCE/'cleaned_monthly.csv'),('computer',SOURCE/'cleaned_computer.csv'),('satisfaction',SOURCE/'cleaned_satisfaction.csv')]
    excluded=[]
    for tag,path in specs:
        files.append(path);table=pd.read_csv(path,float_precision='round_trip');table.date=pd.to_datetime(table.date)
        cols=[c for c in table if c not in ['person','date','source_row','source_rows']]
        if tag=='satisfaction':
            excluded=[c for c in cols if 'Enh/Int' in c or 'Incr/Decr' in c];cols=[c for c in cols if c not in excluded]
        groups={p:g.sort_values('date') for p,g in table.groupby('person')};rows=[]
        for _,r in d.iterrows():
            data=groups.get(r.person);data=data[data.date.le(r.date)] if data is not None else None;rec={}
            for col in cols:
                g=data[data[col].notna()] if data is not None else None;name='current_'+tag+'_'+old.previous.features.slug(col)
                if g is not None and len(g):latest=g.iloc[-1];value=float(latest[col]);age=(r.date-latest.date).days;date=str(latest.date.date())
                else:value=age=np.nan;date=None
                rec[name]=value;rec[name+'_age_days']=age
                audits.append({'source_row':int(r.source_row),'source':tag,'variable':col,'latest_date':date,'bound':str(r.date.date()),'bound_inclusive':True})
            rows.append(rec)
        for col in rows[0]:
            current[col]=[row[col] for row in rows]
            if tag=='satisfaction':environment.append(col)
    path=SOURCE/'fitbit_daily_all.csv';files.append(path);table=pd.read_csv(path,float_precision='round_trip');table.day=pd.to_datetime(table.day);cols=[c for c in table if c.startswith('fit_')];groups={p:g for p,g in table.groupby('person')};rows=[]
    for _,r in d.iterrows():
        g=groups.get(r.person,pd.DataFrame(columns=table.columns));g=g[g.day.ge(r.date-pd.Timedelta(days=8))&g.day.lt(r.date-pd.Timedelta(days=1))];rec={}
        for col in cols:
            v=g[col].dropna();rec['current_'+col+'_week_mean']=float(v.mean()) if len(v)>=3 else np.nan;rec['current_'+col+'_between_day_sd']=float(v.std(ddof=0)) if len(v)>=3 else np.nan
        rows.append(rec);audits.append({'source_row':int(r.source_row),'source':'Fitbit','count':len(g),'latest_date':str(g.day.max().date()) if len(g) else None,'bound':str((r.date-pd.Timedelta(days=1)).date()),'bound_inclusive':False})
    for col in rows[0]:current[col]=[row[col] for row in rows]
    path=SOURCE/'cleaned_diary.csv';files.append(path);table=pd.read_csv(path,float_precision='round_trip');table.date=pd.to_datetime(table.date);table['event_day']=table.date-pd.Timedelta(days=1);cols=[c for c in table if c not in ['person','date','source_rows','event_day']];groups={p:g for p,g in table.groupby('person')};rows=[]
    for _,r in d.iterrows():
        g=groups.get(r.person,pd.DataFrame(columns=table.columns));g=g[g.event_day.ge(r.date-pd.Timedelta(days=8))&g.event_day.lt(r.date-pd.Timedelta(days=1))&g.date.lt(r.date)];rec={}
        for i,col in enumerate(cols,1):
            v=g[col].where(g[col].isin(range(5))).dropna()
            for code in range(5):rec[f'current_diary_item{i}_code{code}_fraction']=float(v.eq(code).mean()) if len(v)>=3 else np.nan
        rows.append(rec);audits.append({'source_row':int(r.source_row),'source':'diary','count':len(g),'latest_date':str(g.date.max().date()) if len(g) else None,'bound':str(r.date.date()),'bound_inclusive':False})
    for col in rows[0]:current[col]=[row[col] for row in rows]
    path=SOURCE/'facial_daily.csv';files.append(path);table=pd.read_csv(path,float_precision='round_trip');table.day=pd.to_datetime(table.day);cols=[c for c in table if c.startswith('face_')];groups={p:g.sort_values('day') for p,g in table.groupby('person')};rows=[]
    for _,r in d.iterrows():
        g=groups.get(r.person,pd.DataFrame(columns=table.columns));g=g[g.day.ge(r.date-pd.Timedelta(days=28))&g.day.lt(r.date)]
        rows.append(g.iloc[-1][cols].to_numpy(float) if len(g) else np.full(len(cols),np.nan))
        audits.append({'source_row':int(r.source_row),'source':'facial','count':len(g),'latest_date':str(g.day.max().date()) if len(g) else None,'bound':str(r.date.date()),'bound_inclusive':False})
    face=pd.DataFrame(rows,columns=['current_'+c for c in cols]);cur=pd.concat([pd.DataFrame(current),face],axis=1)
    prev=cur.groupby(d.person).shift(1);dates=d.groupby('person').date.shift(1);gap=(d.date-dates).dt.days;valid=gap.between(7,21)
    delta_cols=[c for c in cur if not c.startswith('current_face_') and not c.endswith('_age_days')]
    delta=(cur[delta_cols]-prev[delta_cols]).where(valid,axis=0);delta.columns=['change_'+c.removeprefix('current_') for c in delta_cols]
    data=pd.concat([d,cur,delta],axis=1);assert not data.columns.duplicated().any()
    def expanded(cols):return cols+['change_'+c.removeprefix('current_') for c in cols if c in delta_cols]
    h=old.previous.core.OSE_FEATURES
    scopes={'History only':h,'Current physical + changes':h+expanded(physical),'Current environment + changes':h+expanded(physical+environment),'All current levels':h+other+list(cur),'All current + changes':h+other+list(cur)+list(delta)}
    schema['face_features']=schema['face_features']+list(face)
    for name,columns in scopes.items():
        assert len(columns)==len(set(columns));assert not set(columns)&{'OSE','person','student_group','source_row','date','eligible','OSE_change','OSE_after','OSE_before'}
    metadata={'plan':PLAN,'scopes':scopes,'new_columns':list(cur)+list(delta),'physical_current_columns':physical,'environment_current_columns':environment,'delta_source_columns':delta_cols,'excluded_current_learning_impact_items':excluded,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    data.to_csv(OUT/'modeling_features.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'source_timing_audit.csv',index=False)
    (OUT/'feature_schema.json').write_text(json.dumps(schema,indent=2));(OUT/'feature_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Current features ready',len(data),len(cur),len(delta),len(cur.columns),len(delta.columns),flush=True)
    return data,schema,scopes,metadata

if __name__=='__main__':build()
