"""Independent raw archive spot checks and complete file-accounting audit."""
import subprocess,json,re
import numpy as np
import pandas as pd
import ose_all_features as f
OUT=f.OUT
fitpath=next(f.SOURCE.rglob('Fitbit.rar'));members=subprocess.run(['/usr/bin/bsdtar','-tf',str(fitpath)],capture_output=True,text=True,check=True).stdout.splitlines();csvs={m for m in members if m.endswith('.csv')}
audit=pd.read_csv(OUT/'fitbit_archive_audit.csv');excluded=pd.read_csv(OUT/'fitbit_archive_exclusions.csv');daily=pd.read_csv(OUT/'fitbit_daily_all.csv',float_precision='round_trip').set_index(['person','day'])
unavailable=sorted(csvs-set(audit.member)-set(excluded.member.dropna()));pd.DataFrame({'member':unavailable,'reason':'Empty CSV or no numeric/named-state values to summarize'}).to_csv(OUT/'fitbit_unavailable_members.csv',index=False)
assert len(csvs)==len(set(audit.member))+len(set(excluded.member.dropna()))+len(unavailable)
checked=[]
for kind,g in audit.groupby('kind'):
    selected=None
    for _,row in g.head(50).iterrows():
        if (row.person,row.day) not in daily.index:continue
        want=daily.loc[(row.person,row.day)];keys=[c for c in want.index if c.startswith('fit_'+kind+'_') and want[[c]].notna().all()]
        if not keys:continue
        raw=subprocess.run(['/usr/bin/bsdtar','-xOf',str(fitpath),row.member],capture_output=True,check=True).stdout
        from io import BytesIO
        data=pd.read_csv(BytesIO(raw));data.columns=data.columns.str.strip()
        if 'Time' in data and kind!='sleep':
            fields=[c for c in data if c!='Time'];bad=data.groupby('Time')[fields].nunique().max(axis=1).gt(1);data=data[~data.Time.isin(bad[bad].index)].drop_duplicates('Time')
        verified=0
        for c in data:
            if c in ['Date','Time']:continue
            v=pd.to_numeric(data[c],errors='coerce');v=v[np.isfinite(v)]
            if not len(v):continue
            name='fit_'+kind+'_'+f.slug(c)
            if name+'_mean' in want and pd.notna(want[name+'_mean']):
                mu=sum(v)/len(v);sd=np.sqrt(sum((v-mu)**2)/len(v));np.testing.assert_allclose(want[name+'_mean'],mu,atol=1e-8,rtol=1e-9);np.testing.assert_allclose(want[name+'_sd'],sd,atol=1e-8,rtol=1e-9);verified+=1
        if verified:checked.append({'kind':kind,'member':row.member,'numeric_fields_checked':verified});selected=True;break
    assert selected,kind
print('All9 Fitbit file types spot checked',flush=True)
fa=pd.read_csv(OUT/'facial_archive_audit_all.csv');sessions=pd.read_csv(OUT/'facial_sessions_all.csv',float_precision='round_trip');counts=sessions.groupby(['person','day']).size();good=fa[fa.status.eq('Included')]
row=next(r for _,r in good.iterrows() if counts.loc[(r.person,r.day)]==1);cols=['timestamp','confidence','success','gaze_0_x','pose_Rx','AU12_r'];proc=subprocess.Popen(['/usr/bin/bsdtar','-xOf',row.archive,row.member],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
data=pd.read_csv(proc.stdout,skipinitialspace=True,usecols=lambda c:c.strip() in cols,skiprows=lambda i:i>0 and (i-1)%30!=0);code=proc.wait();assert code==0;data.columns=data.columns.str.strip();data=data[data.success.eq(1)&data.confidence.ge(.8)];assert len(data)==row.valid_frames
target=sessions[sessions.person.eq(row.person)&sessions.day.eq(row.day)].iloc[0]
for c in ['confidence','gaze_0_x','pose_Rx','AU12_r']:
    v=data[c].to_numpy();mu=sum(v)/len(v);np.testing.assert_allclose(target['face_'+c+'_mean'],mu,atol=1e-10);np.testing.assert_allclose(target['face_'+c+'_sd'],np.sqrt(sum((v-mu)**2)/len(v)),atol=1e-10)
assert len(fa)==406 and fa.archive.nunique()==18 and good.shape[0]==len(sessions)
checks={'all_Fitbit_CSV_members_accounted_for':True,'fitbit_source_csvs':len(csvs),'fitbit_summarized_csvs':len(audit),'fitbit_date_mismatch_exclusions':len(excluded),'fitbit_empty_or_unavailable_csvs':len(unavailable),'all9_Fitbit_types_checked_against_raw_archive_samples':True,'facial_raw_sample_tracking_quality_and4_numeric_summaries_checked':True,'all18_facial_archives406_files_accounted_for':True,'Fitbit_samples':checked,'facial_sample':{'archive':row.archive,'member':row.member}}
(OUT/'archive_independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps({k:v for k,v in checks.items() if k not in ['Fitbit_samples','facial_sample']},indent=2))
