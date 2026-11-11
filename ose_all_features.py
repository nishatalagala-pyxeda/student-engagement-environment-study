"""All dated predictor sources for later supplied OSE; no outcome-time inputs."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
import ose_stress_prediction as core
ROOT=core.ROOT;SOURCE=core.regression.SOURCE;OUT=ROOT/'outputs/ose-all-variables'

def slug(s):return re.sub('[^A-Za-z0-9]+','_',s).strip('_')

def clean_table(filename,tag,columns=None):
    path=next(SOURCE.rglob(filename));raw=pd.read_csv(path);datecol=next(c for c in raw if 'date' in c.lower());original=list(raw.columns)
    raw['person']=raw.ID.map(core.canonical);raw['date']=pd.to_datetime(raw[datecol],format='mixed',errors='coerce').dt.normalize();raw['source_row']=raw.index+2
    cols=columns if columns is not None else [c for c in original if c not in [datecol,'ID']]
    d=raw[raw.person.notna()&raw.date.notna()].copy()
    for c in cols:d[c]=pd.to_numeric(d[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    g=d.groupby(['person','date']);values=g[cols].first();nunique=g[cols].nunique();values=values.mask(nunique.gt(1));values['source_rows']=g.source_row.apply(lambda v:','.join(map(str,v)))
    values=values.reset_index();values.to_csv(OUT/f'cleaned_{tag}.csv',index=False)
    meta={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_rows':len(raw),'valid_id_date_rows':len(d),'cleaned_person_dates':len(values),'conflicting_person_date_variables_masked':int(nunique.gt(1).sum().sum()),'columns':cols}
    return values,cols,meta

def episodic_sources():
    specs=[('PANAS_PSS.csv','mood',['Positive emotion','Negative emotion']),('CES-D_STAI.csv','monthly',None),('OSE_Computer_Internert.csv','computer',['CPUSpeed_1','InternetSpeed_1','DownloadSpeed','UploadSpeed']),('General_Environmental_Satisfaction_Survey.csv','satisfaction',None)]
    stores={};meta={};groups={}
    for filename,tag,cols in specs:
        data,cols,info=clean_table(filename,tag,cols);stores[tag]={'data':data,'cols':cols,'by_id':{p:g.sort_values('date') for p,g in data.groupby('person')}};meta[tag]=info
        group='psychology' if tag in ['mood','monthly'] else tag;groups.setdefault(group,[]).extend([tag+'_'+slug(c)+'_'+s for c in cols for s in ['last','mean','sd','age_days']])
    return stores,groups,meta

def episodic_features(targets,stores,cutoff=None):
    records=[];audit=[]
    for _,r in targets.iterrows():
        rec={}
        for tag,store in stores.items():
            data=store['by_id'].get(r.person)
            if data is not None:
                data=data[data.date.lt(r.date-pd.Timedelta(days=7))]
                if cutoff is not None:data=data[data.date.lt(cutoff)]
            for col in store['cols']:
                name=tag+'_'+slug(col);g=data[data[col].notna()] if data is not None else None
                if g is not None and len(g):
                    latest=g.iloc[-1];vals=g[col];values=[float(vals.iloc[-1]),float(vals.mean()),float(vals.std(ddof=0)),(r.date-latest.date).days];lastdate=str(latest.date.date());count=len(g)
                    assert latest.date<r.date-pd.Timedelta(days=7) and (cutoff is None or latest.date<cutoff)
                else:values=[np.nan]*4;lastdate=None;count=0
                rec.update({name+'_'+s:v for s,v in zip(['last','mean','sd','age_days'],values)})
                audit.append({'source_row':int(r.source_row),'person':r.person,'target_date':str(r.date.date()),'variable':name,'latest_date':lastdate,'reports':count,'cutoff_exclusive':str(cutoff.date()) if cutoff is not None else None})
        records.append(rec)
    return pd.DataFrame(records),audit

def demographics(d):
    path=next(SOURCE.rglob('*.xlsx'));demo=pd.read_excel(path);demo['person']=demo.ID.map(core.canonical);assert demo.person.is_unique
    demo=demo.set_index('person');r=demo.reindex(d.person).reset_index(drop=True);features={};nums=[];cats=[];excluded=[]
    for col in demo:
        if col=='ID':continue
        if demo[col].notna().sum()==0:excluded.append(col);continue
        name='demo_'+slug(col)
        if col in ['YearBirth','Ladder']:
            val=pd.to_numeric(r[col],errors='coerce')
            features[name]=d.date.dt.year.to_numpy()-val.to_numpy() if col=='YearBirth' else val.to_numpy();nums.append(name)
            if (r[col].notna()&val.isna()).any():
                features[name+'_unparsed']=r[col].where(val.isna()).map(lambda x:str(x) if pd.notna(x) else np.nan);cats.append(name+'_unparsed')
        else:features[name]=r[col].map(lambda x:str(x).strip() if pd.notna(x) else np.nan);cats.append(name)
    return pd.DataFrame(features),nums,cats,{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'excluded_empty_fields':excluded,'note':'All nonempty demographic fields except ID. YearBirth becomes approximate age. PostalCode and open-ended values categorical. Administration date unrecorded; all-without-demographics sensitivity required. No use of student ID as predictor.'}

def sensors(d):
    daily=pd.read_csv(ROOT/'outputs/stress-increases/environment_daily.csv',float_precision='round_trip');daily.day=pd.to_datetime(daily.day)
    # Add the available device composite score, retained as a predictor rather than validated exposure.
    mapping=json.loads((ROOT/'outputs/depressdata-assessment/sensor_id_mapping.json').read_text());mp={v['sensor_id']:v['proposed_person'] for v in mapping if v['proposed_person']};path=next(SOURCE.rglob('Bedroom_IEQ.csv'));raw=pd.read_csv(path);raw['person']=raw.ID.map(mp);raw['day']=pd.to_datetime(raw['timestamp(America/New_York)'],format='mixed').dt.normalize();score=raw.groupby(['person','day']).score.mean().rename('device_score').reset_index();daily=daily.merge(score,on=['person','day'],how='left',validate='one_to_one')
    signals=['temperature_C','humidity','co2','voc','pm25','light','noise','device_score'];names=['env_'+s+'_'+stat for s in signals for stat in ['mean','sd','mean_departure','sd_departure','trend']];records=[];audit=[]
    byid={p:g for p,g in daily.groupby('person')}
    for _,r in d.iterrows():
        start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8);g=byid.get(r.person,pd.DataFrame(columns=daily.columns));recent=g[g.day.ge(start)&g.day.lt(end)];ref=g[g.day.ge(start-pd.Timedelta(days=28))&g.day.lt(start)];row={c:np.nan for c in names}
        if len(recent)>=5:
            x=(recent.day-start).dt.days.to_numpy(float);xc=x-x.mean()
            for s in signals:
                row['env_'+s+'_mean']=recent[s].mean();row['env_'+s+'_sd']=recent[s].std(ddof=0)
                row['env_'+s+'_trend']=float(xc@(recent[s].to_numpy()-recent[s].mean())/(xc@xc)*7)
                if len(ref)>=14:row['env_'+s+'_mean_departure']=recent[s].mean()-ref[s].mean();row['env_'+s+'_sd_departure']=recent[s].std(ddof=0)-ref[s].std(ddof=0)
        records.append(row);audit.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'recent_start':str(start.date()),'recent_end_exclusive':str(end.date()),'reference_start':str((start-pd.Timedelta(days=28)).date()),'recent_days':len(recent),'reference_days':len(ref)})
    pd.DataFrame(audit).to_csv(OUT/'environment_window_audit.csv',index=False);daily.to_csv(OUT/'environment_daily_all.csv',index=False)
    return pd.DataFrame(records),names

def diary(d):
    data,cols,meta=clean_table('Daily_Activity_Diary.csv','diary');data['event_day']=data.date-pd.Timedelta(days=1);byid={p:g for p,g in data.groupby('person')};names=['diary_item'+str(i)+'_code'+str(c)+'_fraction' for i in range(1,len(cols)+1) for c in range(5)];records=[];audit=[]
    for _,r in d.iterrows():
        start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8);g=byid.get(r.person,pd.DataFrame(columns=data.columns));g=g[g.event_day.ge(start)&g.event_day.lt(end)&g.date.lt(r.date-pd.Timedelta(days=7))];row={c:np.nan for c in names}
        for i,col in enumerate(cols,1):
            valid=g[col].where(g[col].isin(range(5))).dropna()
            if len(valid)>=3:
                for c in range(5):row['diary_item'+str(i)+'_code'+str(c)+'_fraction']=float(valid.eq(c).mean())
        records.append(row);audit.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'event_start':str(start.date()),'event_end_exclusive':str(end.date()),'available_report_dates':','.join(pd.to_datetime(g.date).dt.strftime('%Y-%m-%d'))})
    pd.DataFrame(audit).to_csv(OUT/'diary_window_audit.csv',index=False);meta['item_mapping']={str(i):c for i,c in enumerate(cols,1)};meta['note']='Yesterday event day=report date minus1. Separate proportions of recorded codes0..4, not hours. Need3 valid prior diary dates/item. Conflicting same-person/date variable codes excluded, no averaging disagreeing submissions.'
    return pd.DataFrame(records),names,meta

def wearable_features(d):
    daily=pd.read_csv(OUT/'fitbit_daily_all.csv',float_precision='round_trip');daily.day=pd.to_datetime(daily.day);cols=[c for c in daily if c.startswith('fit_')];byid={p:g for p,g in daily.groupby('person')};names=[c+'_'+s for c in cols for s in ['week_mean','between_day_sd']];rows=[];audit=[]
    for _,r in d.iterrows():
        start=r.date-pd.Timedelta(days=15);end=r.date-pd.Timedelta(days=8);g=byid.get(r.person,pd.DataFrame(columns=daily.columns));g=g[g.day.ge(start)&g.day.lt(end)];row={}
        for c in cols:
            v=g[c].dropna();row[c+'_week_mean']=float(v.mean()) if len(v)>=3 else np.nan;row[c+'_between_day_sd']=float(v.std(ddof=0)) if len(v)>=3 else np.nan
        rows.append(row);audit.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'start':str(start.date()),'end_exclusive':str(end.date()),'available_days':','.join(pd.to_datetime(g.day).dt.strftime('%Y-%m-%d'))})
    pd.DataFrame(audit).to_csv(OUT/'fitbit_window_audit.csv',index=False);return pd.DataFrame(rows),names

def facial_features(d):
    daily=pd.read_csv(OUT/'facial_daily.csv',float_precision='round_trip');daily.day=pd.to_datetime(daily.day);cols=[c for c in daily if c.startswith('face_')];byid={p:g.sort_values('day') for p,g in daily.groupby('person')};rows=[];audit=[]
    for _,r in d.iterrows():
        end=r.date-pd.Timedelta(days=8);start=end-pd.Timedelta(days=28);g=byid.get(r.person,pd.DataFrame(columns=daily.columns));g=g[g.day.ge(start)&g.day.lt(end)]
        if len(g):row=g.iloc[-1][cols].to_dict();day=g.iloc[-1].day
        else:row={c:np.nan for c in cols};day=None
        rows.append(row);audit.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'start':str(start.date()),'end_exclusive':str(end.date()),'selected_session_day':str(day.date()) if day is not None else None})
    pd.DataFrame(audit).to_csv(OUT/'facial_window_audit.csv',index=False);return pd.DataFrame(rows),cols

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);ose=pd.read_csv(ROOT/'outputs/ose-stress-prediction/cleaned_OSE.csv',float_precision='round_trip');ose.date=pd.to_datetime(ose.date);stress=pd.read_csv(ROOT/'outputs/ose-stress-prediction/cleaned_PSS.csv',float_precision='round_trip');stress.date=pd.to_datetime(stress.date)
    base=core.history_features(ose,ose,stress);stores,episodic_groups,source_meta=episodic_sources();episodic,audit=episodic_features(ose,stores);pd.DataFrame(audit).to_csv(OUT/'full_episodic_history_audit.csv',index=False)
    demo,dnums,dcats,dmeta=demographics(ose);env,enums=sensors(ose);diary_data,diarynums,diarymeta=diary(ose);fit,fitnums=wearable_features(ose);face,facenums=facial_features(ose)
    all_data=pd.concat([base,episodic,demo,env,diary_data,fit,face],axis=1);assert not all_data.columns.duplicated().any()
    groups={'psychology':core.STRESS_FEATURES+episodic_groups['psychology'],'demographics_time':dnums+dcats+['annual_sin','annual_cos','study_cohort'],'computer':episodic_groups['computer'],'environment':enums+episodic_groups['satisfaction'],'diary':diarynums,'Fitbit':fitnums,'facial':facenums};cats=dcats+['study_cohort']
    all_data.to_csv(OUT/'modeling_features.csv',index=False);schema={'groups':groups,'categorical_features':cats,'face_features':facenums,'episodic_features':[c for g in episodic_groups.values() for c in g],'source_metadata':{**source_meta,'demographics':dmeta,'diary':diarymeta},'coverage':{k:{'reports_any_feature':int(all_data[cs].notna().any(axis=1).sum()),'ids_any_feature':int(all_data.loc[all_data[cs].notna().any(axis=1),'person'].nunique()),'feature_count':len(cs)} for k,cs in groups.items()}}
    (OUT/'feature_schema.json').write_text(json.dumps(schema,indent=2));return ose,stress,all_data,stores,schema

if __name__=='__main__':
    a,b,d,s,m=prepare();print(json.dumps(m['coverage'],indent=2))
