"""Stream Fitbit/facial archives into dated summaries without full extraction."""
from pathlib import Path
import subprocess,tarfile,io,re,json,time,hashlib
import numpy as np
import pandas as pd
from assess_depressdata import canonical,SOURCE
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/ose-all-variables';OUT.mkdir(parents=True,exist_ok=True)

class NonSeekable(io.RawIOBase):
    def __init__(self,raw):self.raw=raw
    def readable(self):return True
    def seekable(self):return False
    def readinto(self,b):
        data=self.raw.read(len(b));b[:len(data)]=data;return len(data)

def csv_input(raw):return io.BufferedReader(NonSeekable(raw),buffer_size=1048576)

def archive_stream(path):
    # libarchive transcodes RAR to streaming TAR; no raw archive contents written to disk.
    proc=subprocess.Popen(['/usr/bin/bsdtar','-cf','-','@'+str(path)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    return proc,tarfile.open(fileobj=proc.stdout,mode='r|')

def fitbit():
    path=next(SOURCE.rglob('Fitbit.rar'));proc,archive=archive_stream(path);records=[];audit=[];fail=[]
    for member in archive:
        if not member.isfile() or not member.name.endswith('.csv'):continue
        match=re.search(r'(\d{8})\.csv$',member.name)
        if not match:continue
        day=pd.to_datetime(match[1],format='%Y%m%d',errors='coerce');person=canonical(member.name.split('/')[1]);kind=re.sub(r'\d.*','',Path(member.name).stem)
        try:
            d=pd.read_csv(csv_input(archive.extractfile(member)));d.columns=d.columns.str.strip()
            if d.empty:continue
            if 'Date' in d:
                actual=pd.to_datetime(d.Date,format='mixed',errors='coerce').dt.normalize()
                if not actual.eq(day).all():fail.append({'member':member.name,'reason':'CSV Date disagrees with filename'});continue
            row={'person':person,'day':day,'kind':kind};cols=[]
            if 'Time' in d and kind!='sleep':
                # Intraday streams are local clock values; collapse duplicate times only if equal.
                non_time=[c for c in d if c!='Time'];conflict=d.groupby('Time')[non_time].transform('nunique').gt(1).any(axis=1)
                d=d[~conflict].drop_duplicates('Time')
            for c in d:
                if c in ['Date','Time']:continue
                vals=pd.to_numeric(d[c],errors='coerce');vals=vals[np.isfinite(vals)]
                if len(vals):
                    key='fit_'+kind+'_'+re.sub('[^A-Za-z0-9]+','_',c).strip('_')
                    row[key+'_mean']=float(vals.mean());row[key+'_sd']=float(vals.std(ddof=0));cols.extend([key+'_mean',key+'_sd'])
                elif c in ['Interpreted','name']:
                    for label,n in d[c].fillna('Missing').astype(str).value_counts().items():
                        key='fit_'+kind+'_'+c+'_'+re.sub('[^A-Za-z0-9]+','_',label).strip('_')+'_fraction';row[key]=float(n/len(d));cols.append(key)
            if not cols:continue
            records.append(row);audit.append({'member':member.name,'person':person,'day':str(day.date()),'kind':kind,'rows':len(d)})
            if len(audit)%3000==0:print('Fitbit files',len(audit),flush=True)
        except Exception as e:fail.append({'member':member.name,'reason':type(e).__name__+': '+str(e)[:180]})
    archive.close();proc.stdout.close();code=proc.wait();assert code==0,code
    raw=pd.DataFrame(records);clean=[];conflicts=[]
    for (person,day,kind),g in raw.groupby(['person','day','kind']):
        cols=[c for c in raw if c.startswith('fit_') and g[c].notna().any()]
        if len(g.drop_duplicates(cols))>1:conflicts.append({'person':person,'day':str(day.date()),'kind':kind,'exports':len(g)});continue
        clean.append(g.iloc[0])
    long=pd.DataFrame(clean);daily=long.groupby(['person','day'],as_index=False).first().drop(columns='kind');daily.to_csv(OUT/'fitbit_daily_all.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'fitbit_archive_audit.csv',index=False);pd.DataFrame(fail+conflicts).to_csv(OUT/'fitbit_archive_exclusions.csv',index=False)
    (OUT/'fitbit_archive_metadata.json').write_text(json.dumps({'source':str(path),'bytes':path.stat().st_size,'files_summarized':len(audit),'excluded_files_or_conflicting_exports':len(fail)+len(conflicts),'daily_rows':len(daily),'ids':int(daily.person.nunique()),'features':[c for c in daily if c.startswith('fit_')],'notes':'All9 file types summarized. Raw native values/units; no clinical interpretation or verified wear. Means/SDs within file, category fractions for named states; repeated conflicting daily exports excluded. Sleep intraday clock resets can cross midnight: assigned to filename date (documented collection date), no sleep-hours conversion. Heart sample means not time-weighted. CSV Date mismatches excluded. Earlier feature window has1day timezone buffer.'},indent=2));print('Fitbit complete',len(daily),'daily rows',flush=True)

def faces(worker=None,workers=1):
    global OUT
    paths=sorted(SOURCE.rglob('Facial_expression_*.rar'))
    if worker is not None:
        checkpoint=OUT/'facial_archive_audit.csv';completed=set(pd.read_csv(checkpoint).archive) if checkpoint.exists() else set()
        paths=[p for p in paths if str(p) not in completed];paths=[p for i,p in enumerate(paths) if i%workers==worker]
        OUT=OUT/f'facial-worker-{worker}';OUT.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(ROOT/'outputs/ose-stress-prediction/cleaned_OSE.csv');labels.date=pd.to_datetime(labels.date);limits=labels.groupby('person').date.agg(['min','max']);records=[];audit=[]
    for path in paths:
        proc,archive=archive_stream(path);n=0
        for member in archive:
            if not member.isfile() or not member.name.endswith('.csv'):continue
            parts=member.name.split('/');person=canonical(parts[-2]);match=re.search(r'[-_](\d{8})',Path(member.name).name)
            day=pd.to_datetime(match[1],format='%d%m%Y',errors='coerce') if match else pd.NaT
            rec={'archive':str(path),'member':member.name,'person':person,'day':str(day.date()) if pd.notna(day) else None,'archive_member_bytes':member.size}
            if person not in limits.index or pd.isna(day) or day<limits.loc[person,'min']-pd.Timedelta(days=14) or day>limits.loc[person,'max']:
                audit.append({**rec,'status':'Excluded: unknown person/date or outside participant survey period ±14days before first survey'});continue
            try:
                f=archive.extractfile(member);d=pd.read_csv(csv_input(f),skipinitialspace=True,skiprows=lambda i:i>0 and (i-1)%30!=0)
                d.columns=d.columns.str.strip();n+=1
                good=d[d.success.eq(1)&d.confidence.ge(.8)].copy();span=float(good.timestamp.max()-good.timestamp.min()) if len(good) else 0.
                if len(good)<30 or span<60:audit.append({**rec,'status':'Excluded: fewer than30 successful sampled frames or span<60sec','sampled_frames':len(d),'valid_frames':len(good)});continue
                row={'person':person,'day':day}
                for col in d:
                    if col in ['frame','face_id','timestamp','success']:continue
                    values=pd.to_numeric(good[col],errors='coerce');values=values[np.isfinite(values)]
                    if len(values):row['face_'+col+'_mean']=float(values.mean());row['face_'+col+'_sd']=float(values.std(ddof=0))
                records.append(row);audit.append({**rec,'status':'Included','sampled_frames':len(d),'valid_frames':len(good),'sampled_span_seconds':span});print('Face summarized',person,str(day.date()),len(good),'frames',flush=True)
            except Exception as e:audit.append({**rec,'status':'Parse failure: '+type(e).__name__+': '+str(e)[:180]})
        archive.close();proc.stdout.close();code=proc.wait();assert code==0,(path,code)
        pd.DataFrame(audit).to_csv(OUT/'facial_archive_audit.csv',index=False);pd.DataFrame(records).to_csv(OUT/'facial_sessions.csv',index=False)
        print('Facial archive complete',path.name,n,'parsed files',flush=True)
    data=pd.DataFrame(records);assert not data.empty
    # Multiple valid sessions on a date are averaged equally, never concatenated as simultaneous frames.
    daily=data.groupby(['person','day'],as_index=False).mean(numeric_only=True);daily.to_csv(OUT/'facial_daily.csv',index=False)
    (OUT/'facial_archive_metadata.json').write_text(json.dumps({'archive_count':len(paths),'files_audited':len(audit),'valid_sessions':len(data),'daily_rows':len(daily),'ids':int(daily.person.nunique()),'features':[c for c in daily if c.startswith('face_')],'notes':'All numeric tracking/gaze/pose/landmark/AU columns summarized except frame ID, face ID, timestamp and success. Deterministic every30th row sampled before parsing, require success1/confidence>=0.8,30valid sampled frames and60sec span. Filename dates strictlyDDMMYYYY per README; unknown/implausible dates excluded rather than guessed. Sampling is by row, not assumed fps. Camera geometry not normalized; not independent emotion labels. Train-only dimension reduction is required in modeling.'},indent=2));print('All face summaries complete',len(daily),flush=True)

def merge_faces():
    paths=[OUT]+sorted(OUT.glob('facial-worker-*'));sessions=pd.concat([pd.read_csv(p/'facial_sessions.csv') for p in paths if (p/'facial_sessions.csv').exists()],ignore_index=True);audit=pd.concat([pd.read_csv(p/'facial_archive_audit.csv') for p in paths if (p/'facial_archive_audit.csv').exists()],ignore_index=True)
    assert audit[['archive','member']].duplicated().sum()==0
    assert len(audit.archive.unique())==len(list(SOURCE.rglob('Facial_expression_*.rar')))
    assert len(sessions)==audit.status.eq('Included').sum()
    def filename_person(member):
        m=re.match(r'([A-Za-z]+)[_ ]*(\d+)[-_]',Path(member).name)
        return canonical(m[1]+' '+m[2]) if m else None
    audit['filename_person']=audit.member.map(filename_person)
    bad=audit[audit.status.eq('Included')&audit.filename_person.ne(audit.person)]
    badkeys=set(map(tuple,bad[['person','day']].to_numpy()))
    if badkeys:
        take=[(p,day) not in badkeys for p,day in zip(sessions.person,sessions.day)];sessions=sessions[take].copy()
        bad_audit=[(p,day) in badkeys for p,day in zip(audit.person,audit.day)];audit.loc[np.array(bad_audit)&audit.status.eq('Included'),'status']='Excluded: filename participant disagrees with folder or cannot be parsed'
    assert len(sessions)==audit.status.eq('Included').sum()
    daily=sessions.groupby(['person','day'],as_index=False).mean(numeric_only=True);daily.to_csv(OUT/'facial_daily.csv',index=False);sessions.to_csv(OUT/'facial_sessions_all.csv',index=False);audit.to_csv(OUT/'facial_archive_audit_all.csv',index=False)
    (OUT/'facial_archive_metadata.json').write_text(json.dumps({'archive_count':int(audit.archive.nunique()),'files_audited':len(audit),'valid_sessions':len(sessions),'daily_rows':len(daily),'ids':int(daily.person.nunique()),'features':[c for c in daily if c.startswith('face_')],'status_counts':audit.status.value_counts().to_dict(),'notes':'Every30th source row sampled; all numeric OpenFace columns except frame/face ID,timestamp,success. Confidence>=0.8/success1,>=30 valid sampled rows and60sec span; DDMMYYYY filenames and plausible participant survey dates required. Means/SDs, equal session averages per day; no emotion labels. Root checkpoint plus disjoint worker archives merged with exact membership checks.'},indent=2));print('Merged faces',len(daily),'days',flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='fitbit':fitbit()
    elif sys.argv[1]=='mergefaces':merge_faces()
    else:faces(int(sys.argv[2]) if len(sys.argv)>2 else None,int(sys.argv[3]) if len(sys.argv)>3 else 1)
