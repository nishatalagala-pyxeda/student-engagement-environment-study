"""Quality-flagged raw wearable features, then tabular-plus-wearable comparisons."""
from pathlib import Path
import os,json,hashlib,html,zipfile
os.environ.setdefault('JOBLIB_MULTIPROCESSING','0')
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
import joblib
from sklearn.base import clone
import broad_environment_model as cls
import valence_regression as reg
import all_predictors as tabular
import demographics_step1 as step1

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/all-predictors-with-wearables'
RAW=Path('/Users/nishatalagala/Downloads/Figshare-Navya/Raw_wearable_data.zip')
WINDOW=1800
MIN_SECONDS=300

def read_uniform(z,name):
    if not name or z.getinfo(name).file_size==0:return None
    raw=z.read(name);parts=raw.split(b'\n',2)
    if len(parts)<3:return None
    start=float(parts[0].split(b',')[0]);hz=float(parts[1].split(b',')[0])
    values=np.fromstring(parts[2].replace(b',',b' ').decode('ascii'),sep=' ')
    channels=3 if name.endswith('/ACC.csv') else 1
    if values.size%channels:raise ValueError('Malformed signal rows: '+name)
    values=values.reshape(-1,channels)
    return start,hz,values

def read_ibi(z,name):
    if not name or z.getinfo(name).file_size==0:return None
    raw=z.read(name);parts=raw.split(b'\n',1)
    if len(parts)<2:return None
    start=float(parts[0].split(b',')[0]);clean=[]
    for line in parts[1].splitlines():
        if not line.strip():continue
        cells=line.split(b',')
        if len(cells)==2 and cells[1].strip()==b'IBI':
            if float(cells[0])!=start:raise ValueError('Changed IBI time origin inside '+name)
            continue  # Repeated identical start header is metadata, not a heartbeat.
        if len(cells)!=2:raise ValueError('Malformed IBI data row in '+name)
        clean.append((float(cells[0]),float(cells[1])))
    values=np.asarray(clean,dtype=float).reshape(-1)
    if len(values)%2:raise ValueError('Malformed IBI rows: '+name)
    return start,values.reshape(-1,2)

def moments(values,prefix):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if len(x)<2:return {prefix+'_'+n:np.nan for n in ['mean','std','p10','p90']}
    return {prefix+'_mean':float(x.mean()),prefix+'_std':float(x.std(ddof=1)),prefix+'_p10':float(np.quantile(x,.1)),prefix+'_p90':float(np.quantile(x,.9))}

def uniform_features(record,window_start,window_end,signal):
    prefix='wear_'+signal.lower();out={prefix+'_coverage':0.,prefix+'_nonfinite_fraction':np.nan}
    empty= moments([],prefix+'_magnitude' if signal=='ACC' else prefix)
    if signal=='ACC':empty.update({'wear_acc_step_change':np.nan,'wear_acc_saturation_fraction':np.nan})
    if signal=='EDA':empty.update({'wear_eda_zero_fraction':np.nan,'wear_eda_positive_coverage':0.})
    if signal=='TEMP':empty['wear_temp_slope_per_minute']=np.nan
    if not record:return {**out,**empty}
    start,hz,a=record;left=max(0,int(np.ceil((window_start-start)*hz-1e-6)));right=min(len(a),int(np.ceil((window_end-start)*hz-1e-6)))
    if right<=left:return {**out,**empty}
    v=a[left:right];sample_times=start+np.arange(left,right)/hz
    assert (sample_times>=window_start-1e-6).all() and (sample_times<window_end+1e-6).all()
    finite=np.isfinite(v).all(axis=1);out[prefix+'_coverage']=len(v)/(WINDOW*hz);out[prefix+'_nonfinite_fraction']=float(1-finite.mean())
    out.update(empty)
    if signal=='EDA':
        x=v[:,0];out['wear_eda_zero_fraction']=float(np.mean(x[finite]<=0)) if finite.any() else np.nan
        positive=finite&(x>0);out['wear_eda_positive_coverage']=float(positive.sum()/(WINDOW*hz))
        # Filter a contiguous recording window, without concatenating different sessions.
        smoothed=median_filter(np.where(np.isfinite(x),x,0.),size=max(1,int(5*hz)),mode='nearest');good=positive&(smoothed>0)
        if good.sum()>=MIN_SECONDS*hz:out.update(moments(smoothed[good],prefix))
    elif signal=='ACC':
        out['wear_acc_saturation_fraction']=float((np.abs(v[finite])>=127).any(axis=1).mean()) if finite.any() else np.nan
        if finite.sum()>=MIN_SECONDS*hz:
            g=v[finite]/64.;magnitude=np.linalg.norm(g,axis=1);out.update(moments(magnitude,prefix+'_magnitude'))
            adjacent=finite[1:]&finite[:-1];allmag=np.linalg.norm(v/64.,axis=1);out['wear_acc_step_change']=float(np.mean(np.abs(np.diff(allmag))[adjacent])) if adjacent.any() else np.nan
    else:
        good=finite
        if signal=='HR':good=good&(v[:,0]>0)
        if good.sum()>=MIN_SECONDS*hz:
            x=v[good,0];out.update(moments(x,prefix))
            if signal=='TEMP':out['wear_temp_slope_per_minute']=float(np.polyfit((sample_times[good]-sample_times[good][0])/60,x,1)[0])
    return out

def ibi_features(record,left,right):
    out={'wear_ibi_event_count':0.,'wear_ibi_mean_ms':np.nan,'wear_ibi_std_ms':np.nan,'wear_ibi_rmssd_ms':np.nan,'wear_ibi_adjacent_pair_count':0.,'wear_ibi_duplicate_events':0.,'wear_ibi_conflicting_events':0.}
    if not record:return out
    start,a=record;t=start+a[:,0];valid=np.isfinite(t)&np.isfinite(a[:,1])&(a[:,1]>0)&(t>=left)&(t<right);t=t[valid];v=a[valid,1]
    if not len(v):return out
    unique=np.unique(np.column_stack([t,v]),axis=0);out['wear_ibi_duplicate_events']=float(len(v)-len(unique));t=unique[:,0];v=unique[:,1]
    _,inverse,counts=np.unique(t,return_inverse=True,return_counts=True);conflict=counts[inverse]>1;out['wear_ibi_conflicting_events']=float(conflict.sum());t=t[~conflict];v=v[~conflict]
    if not len(v):return out
    order=np.argsort(t,kind='stable');t=t[order];v=v[order];out['wear_ibi_event_count']=float(len(v))
    if len(v)>=30:out['wear_ibi_mean_ms']=float(v.mean()*1000);out['wear_ibi_std_ms']=float(v.std(ddof=1)*1000)
    adjacent=np.isclose(np.diff(t),v[1:],rtol=.1,atol=.05)&(np.diff(t)>0);out['wear_ibi_adjacent_pair_count']=float(adjacent.sum())
    if adjacent.sum()>=30:out['wear_ibi_rmssd_ms']=float(np.sqrt(np.mean(np.diff(v)[adjacent]**2))*1000)
    return out

def extract(df):
    with zipfile.ZipFile(RAW) as z:
        names=set(z.namelist());sessions=[];hr_cache={};header_issues=[]
        # HR is only 1 Hz: its exact sample count provides a cheap recording duration.
        for name in sorted(n for n in names if n.endswith('/HR.csv')):
            session=name.rsplit('/',1)[0];pid=session.rsplit('_',1)[-1]
            if not pid.isdigit() or int(pid) not in set(df.Pid):continue
            hr=read_uniform(z,name)
            if not hr or not len(hr[2]):continue
            hr_cache[session]=hr;acc_name=session+'/ACC.csv'
            if acc_name not in names:header_issues.append({'session':session,'issue':'missing ACC start header'});continue
            with z.open(acc_name) as stream:start=float(stream.readline().split(b',')[0])
            end=hr[0]+len(hr[2])/hr[1]
            if end<=start:header_issues.append({'session':session,'issue':'nonpositive recording span'});continue
            sessions.append({'pid':int(pid),'session':session,'start':start,'end':end})
        assignments=[];groups={};feature_rows={}
        for _,row in df.iterrows():
            end=pd.Timestamp(row.timestamp).tz_localize('Australia/Melbourne').tz_convert('UTC').timestamp();left=end-WINDOW
            candidates=[]
            for session in sessions:
                if session['pid']!=row.Pid:continue
                coverage=max(0,min(end,session['end'])-max(left,session['start']))
                if coverage>0:candidates.append((coverage,session['start'],session))
            selected=max(candidates,key=lambda x:(x[0],x[1],x[2]['session']))[2] if candidates else None
            rec={'source_row':int(row.source_row),'wear_recording_coverage':max([c[0] for c in candidates],default=0)/WINDOW,'wear_multiple_recordings':int(len(candidates)>1)}
            info={'source_row':int(row.source_row),'participant':int(row.Pid),'date':row.date,'window_end_local':row.timestamp.isoformat(),'window_start_utc':pd.to_datetime(left,unit='s',utc=True).isoformat(),'window_end_utc':pd.to_datetime(end,unit='s',utc=True).isoformat(),'candidate_recordings':len(candidates),'selected_recording':selected['session'] if selected else None}
            assignments.append(info);feature_rows[int(row.source_row)]=rec
            if selected:groups.setdefault(selected['session'],[]).append((int(row.source_row),left,end))
            else:
                for signal in ['ACC','BVP','EDA','HR','TEMP']:rec.update(uniform_features(None,left,end,signal))
                rec.update(ibi_features(None,left,end))
        for i,(session,windows) in enumerate(sorted(groups.items()),1):
            for signal in ['ACC','BVP','EDA','HR','TEMP','IBI']:
                name=session+'/'+signal+'.csv';name=name if name in names else None
                if signal=='IBI':record=read_ibi(z,name)
                elif signal=='HR':record=hr_cache.get(session)
                else:record=read_uniform(z,name)
                for source_row,left,end in windows:
                    features=ibi_features(record,left,end) if signal=='IBI' else uniform_features(record,left,end,signal)
                    feature_rows[source_row].update(features)
                del record
            if i%10==0:print('Extracted recordings',i,'of',len(groups),flush=True)
    features=pd.DataFrame([feature_rows[int(i)] for i in df.source_row]);assert features.source_row.is_unique
    assignments=pd.DataFrame(assignments)
    metadata={'window_seconds':WINDOW,'minimum_uniform_valid_seconds_for_summary':MIN_SECONDS,'timezone':'UTC raw Unix times → Australia/Melbourne with DST; original class notebook fixed UTC+10 is not reused','recording_policy':'Choose one recording per survey: greatest pre-survey window coverage inferred from exact HR length and ACC start; ties use latest recording start. Never concatenate overlapping recordings. HR-derived span is a selection proxy; each modality computes its own actual sample coverage.','selected_recordings':len(groups),'responses_with_candidate':int(assignments.candidate_recordings.gt(0).sum()),'responses_with_multiple_candidates':int(assignments.candidate_recordings.gt(1).sum()),'header_issues':header_issues,'coverage_by_signal':{s:features['wear_'+s.lower()+'_coverage'].describe().to_dict() for s in ['ACC','BVP','EDA','HR','TEMP']},'missing_by_feature':features.drop(columns='source_row').isna().sum().to_dict(),'raw_archive_bytes':RAW.stat().st_size,'raw_archive':str(RAW),'noncanonical_signal_files':'ACC1/ACC2/EDA1/EDA2 alternatives are not silently merged; canonical named modalities only.','limitations':'Coverage and zero/saturation checks are not a validated full motion-artifact/off-wrist pipeline. EDA uses a five-second median filter and excludes nonpositive samples from summaries. IBI RMSSD needs 30 adjacent pairs; sparse IBI observations are not continuous HRV ground truth. Source clocks have no independent event-anchor validation.'}
    return features,assignments,metadata

def specs_for(task,oldscopes,wear):
    old=tabular.specs_for(task,oldscopes);baseline=['Majority baseline'] if task=='classification' else ['Mean baseline','Median baseline']
    chosen=['All context','All non-mood tabular','Same-PAM diagnostic'];result={name:old[name] for name in baseline}
    for scope in chosen:
        nums,cats=oldscopes[scope]
        for forest,label in [(False,'linear'),(True,'forest')]:
            name=scope+' | '+label;result[name]=old[name]
            result[scope+' + wearables | '+label]=(nums+wear+cats,tabular.build_model(task,nums+wear,cats,forest))
    for forest,label in [(False,'linear'),(True,'forest')]:result['Wearables only | '+label]=(wear,tabular.build_model(task,wear,[],forest))
    return result

def report(output):
    def table(task,protocol):
        keys=['accuracy','balanced_accuracy','roc_auc'] if task=='classification' else ['mae','rmse','r2']
        names=['Accuracy ↑','Balanced accuracy ↑','ROC AUC ↑'] if task=='classification' else ['MAE ↓','RMSE ↓','R² ↑']
        rows=''.join('<tr><td>'+html.escape(n)+'</td>'+''.join(f'<td>{m[k]:.3f}</td>' for k in keys)+'</tr>' for n,m in output['evaluation'][task][protocol].items())
        return '<table><tr><th>Inputs/model</th>'+''.join('<th>'+n+'</th>' for n in names)+'</tr>'+rows+'</table>'
    meta=output['extraction'];m=output['metadata'];head='''<!doctype html><html><head><meta charset="utf-8"><title>All predictors including wearables</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}code{overflow-wrap:anywhere}</style></head><body><h1>All available predictors, including raw wearables</h1><p>Exploratory valence classification and regression; November 10, 2026.</p>'''
    head+=f'''<p>Same 163 responses, 22 students and nine dates. Added {len(m['wearable_features'])} wearable signal/quality fields; raw recordings overlap {meta['responses_with_candidate']} responses, with {meta['responses_with_multiple_candidates']} having multiple candidate sessions. All rows are retained; missing signal summaries use training-only imputation and missing indicators. Models can therefore learn signal availability as well as physiological values.</p>
    <h2>What “all predictors” includes</h2><p>All available table fields were compared first: demographics; three thermal-background answers; form/maths/language groups; subject/room/class schedule context; clock/weekday/study week; all measured outdoor fields; fixed-eight-room campus environmental/state/occupancy proxies; current comfort, clothing, seating, five engagement items and confidence. See the <a href="../all-tabular-predictors/report.html">complete table-only comparison</a> and feature catalog.</p><p>Wearable inputs now cover all six canonical modalities: ACC magnitude/movement and saturation, BVP amplitude summaries, positive EDA after five-second median filtering, HR, skin temperature and trend, and event-based IBI mean/std/RMSSD where enough adjacent intervals exist. Coverage, nonfinite/zero fractions, recording ambiguity and IBI event/pair counts are retained as quality indicators. We do not recompute every conceivable engineered feature or claim full clinical-grade artifact correction.</p>
    <h2>Timing and overlap handling</h2><p>Use the original Figshare raw Unix timestamps converted from UTC to Australia/Melbourne, with daylight saving handled explicitly. Every feature window is [survey − 30 minutes, survey), with no samples at/after submission. This avoids the author notebook's fixed UTC+10 conversion for the later block. Recordings are selected independently of valence, using greatest available pre-survey coverage and latest-start tie-break. Exactly one session is used per response; recordings are never concatenated. Each modality computes its own actual sample coverage. Missing or inadequate signals stay missing rather than being fabricated.</p><p>Uniform-signal summaries require at least five minutes of eligible data in the 30-minute window. EDA flat/nonpositive samples are counted and excluded from filtered-value summaries. ACC is converted from 1/64 g to g. IBI mean/std require 30 events, and RMSSD requires 30 adjacent pairs with elapsed time consistent with the next reported interval (10% relative plus 50 ms absolute tolerance); gaps are not bridged. This remains exploratory signal engineering, not validated wear detection or accurate clinical HRV.</p>
    <h2>Different levels of outcome proximity</h2><ul><li><strong>All context + wearables:</strong> background, schedule and sensor signals; no current survey answers. This is the appropriate independent-context comparison.</li><li><strong>All non-mood tabular + wearables:</strong> additionally uses contemporaneous comfort, engagement, clothing, seating and confidence; a questionnaire-assisted estimate, not an environment-only or future-mood forecast.</li><li><strong>Same-PAM diagnostic + wearables:</strong> also uses arousal derived from the very image choice that produces valence. Kept separate; any gain is not evidence of independent emotional-state prediction.</li></ul><p>Valence, its binary encoding, participant/class/session identifiers, source row and exact calendar date are never predictors. Teacher metadata cannot be linked reliably to these responses and is not invented.</p>
    <h2>Evaluation</h2><p>Same held-out-date, October-to-November and held-out-student protocols as before, with training-only scaling/imputation/encoding. Baseline predictions are reproduced from the prior table-only runs. Logistic C=1, ridge alpha=1, and 300-tree depth-3/minimum-leaf-10 forests are unchanged; no tuning search. Regression is evaluated after clipping to 1–4. Nine dates remain the limiting number of independent environmental contexts.</p>'''
    ev=output['evaluation'];c=ev['classification'];r=ev['regression'];base='All context | forest';wear='All context + wearables | forest'
    head+=f'''<h2>What changed with wearables</h2><p>For the fixed context random forest, held-out-date accuracy changes from {c['held_out_date'][base]['accuracy']:.1%} to {c['held_out_date'][wear]['accuracy']:.1%}; balanced accuracy changes from {c['held_out_date'][base]['balanced_accuracy']:.1%} to {c['held_out_date'][wear]['balanced_accuracy']:.1%}. The paired date-cluster interval for the accuracy change includes zero. On the later November block, accuracy is {c['future_block'][base]['accuracy']:.1%} without and {c['future_block'][wear]['accuracy']:.1%} with wearables, versus {c['future_block']['Majority baseline']['accuracy']:.1%} for the majority baseline. There is no consistent evidence of improvement.</p><p>Context regression MAE changes from {r['held_out_date'][base]['mae']:.3f} to {r['held_out_date'][wear]['mae']:.3f} on held-out dates (lower is better), compared with {r['held_out_date']['Median baseline']['mae']:.3f} for the median baseline. Its later-block MAE changes from {r['future_block'][base]['mae']:.3f} to {r['future_block'][wear]['mae']:.3f}. Adding more inputs has not established reliable valence prediction.</p><p>One selected IBI file contains a repeated identical timestamp header inside its body. The parser skips identical-origin repeated headers and rejects changed origins. Repeated identical beat events are deduplicated within each selected window; conflicting same-time events are excluded. The recording audit records the resulting counts.</p>'''
    for protocol in ['held_out_date','future_block','held_out_student']:
        for task in ['classification','regression']:head+=f'<h2>{protocol}: {task}</h2>'+table(task,protocol)
    head+='''<h2>Limitations and outputs</h2><p>These are sequential exploratory experiments after earlier scores were inspected, not an untouched final test. Missingness, participant calibration, repeated students, timetable gaps, campus rather than individual-room exposure, and source clock/artifact uncertainty remain. A model may exploit whether a wristband was recording; that is not necessarily a physiological effect. Paired intervals resample fixed date/student prediction clusters without refitting and do not capture full crossed-dependency or training uncertainty. No causal or clinical claim is made.</p><p><a href="metrics.json">Metrics and paired comparisons</a>; <a href="wearable_extraction.json">signal coverage/quality methods</a>; <a href="feature_catalog.json">full feature catalog</a>; <a href="recording_assignments.csv">recording/window audit</a>; <a href="classification_predictions.csv">held-out classification</a>; <a href="regression_predictions.csv">held-out regression</a>; <a href="modeling_data.csv">prepared research observations</a>. Fitted payloads retain the encoders, imputers, selected input names and bounds. Run wearable_predictors.py to reproduce; original files and earlier analyses are unchanged.</p></body></html>'''
    (DEST/'report.html').write_text(head)

def main():
    DEST.mkdir(parents=True,exist_ok=True);source=ROOT/'outputs/all-tabular-predictors/modeling_data.csv';df=pd.read_csv(source,parse_dates=['timestamp','feature_end'])
    assert len(df)==163 and df.source_row.is_unique
    features,assignments,extraction=extract(df);features.to_csv(DEST/'wearable_features.csv',index=False);assignments.to_csv(DEST/'recording_assignments.csv',index=False);(DEST/'wearable_extraction.json').write_text(json.dumps(extraction,indent=2))
    df=df.merge(features,on='source_row',how='left',validate='one_to_one',sort=False);wear=[c for c in features if c!='source_row'];df.to_csv(DEST/'modeling_data.csv',index=False)
    prior=json.loads((ROOT/'outputs/all-tabular-predictors/metrics.json').read_text());oldscopes={}
    for name,cols in prior['metadata']['scope_features'].items():
        numeric=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])];cats=[c for c in cols if c not in numeric];oldscopes[name]=(numeric,cats)
    # Read study-week as a category exactly as in the prior tabular preparation.
    df['study_week']=df.study_week.astype(str)
    for name,cols in prior['metadata']['scope_features'].items():
        cats=[c for c in cols if c in ['scheduled_subject_context','form_group','math_group','language_group','study_week','scheduled_room','recent_subject','current_thermal_preference']];oldscopes[name]=([c for c in cols if c not in cats],cats)
    catalog=json.loads((ROOT/'outputs/all-tabular-predictors/feature_catalog.json').read_text())
    for c in wear:catalog[c]={'source':'Original raw wearable recording before survey','availability':'Pre-survey wearable signal or quality/availability indicator','meaning':'See wearable_extraction.json for window, units, minimum coverage and session selection.'}
    (DEST/'feature_catalog.json').write_text(json.dumps(catalog,indent=2));results={};intervals={};payloads={}
    for task in ['classification','regression']:
        specs=specs_for(task,oldscopes,wear);results[task]={};parts=[]
        for protocol in ['held_out_date','future_block','held_out_student']:
            scores,p=(cls.run_protocol if task=='classification' else reg.evaluate)(df,specs,protocol);results[task][protocol]=scores;parts.append(p);print('Completed',task,protocol,flush=True)
        pred=pd.concat(parts,ignore_index=True);oldpred=pd.read_csv(ROOT/'outputs/all-tabular-predictors'/(task+'_predictions.csv'));column='probability_higher' if task=='classification' else 'predicted_valence'
        for name in set(pred.model).intersection(oldpred.model):
            a=pred[pred.model.eq(name)].set_index(['protocol','source_row']).sort_index();b=oldpred[oldpred.model.eq(name)].set_index(['protocol','source_row']).sort_index();assert a.index.equals(b.index);np.testing.assert_allclose(a[column],b[column],atol=1e-8,rtol=0)
        pred.to_csv(DEST/(task+'_predictions.csv'),index=False);intervals[task]={}
        for protocol in ['held_out_date','future_block','held_out_student']:
            intervals[task][protocol]={}
            for scope in ['All context','All non-mood tabular','Same-PAM diagnostic']:
                for family in ['linear','forest']:
                    old=scope+' | '+family;new=scope+' + wearables | '+family;v=step1.paired_interval(pred,task,protocol,new,old);v['unit']=v['unit'].replace('demographics','wearable addition');intervals[task][protocol][new]=v
        target='higher_valence' if task=='classification' else 'Valence';fitted={n:clone(model).fit(df[cols],df[target]) for n,(cols,model) in specs.items()}
        payloads[task]={'models':fitted,'features':{n:cols for n,(cols,_) in specs.items()},'bounds':(1,4) if task=='regression' else None,'target':target}
    metadata={'n':len(df),'students':int(df.Pid.nunique()),'dates':int(df.date.nunique()),'wearable_features':wear,'table_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'old_predictions_reproduced':True,'same_PAM_diagnostic_separate':True}
    for task,payload in payloads.items():
        payload['metadata']=metadata;joblib.dump(payload,DEST/(task+'_models.joblib'));restored=joblib.load(DEST/(task+'_models.joblib'));name='All context + wearables | linear';cols=payload['features'][name];np.testing.assert_allclose(restored['models'][name].predict(df[cols]),payload['models'][name].predict(df[cols]))
    output={'metadata':metadata,'extraction':extraction,'evaluation':results,'paired_changes':intervals};(DEST/'metrics.json').write_text(json.dumps(output,indent=2));report(output);print(json.dumps({'metadata':metadata,'coverage':extraction['responses_with_candidate'],'evaluation':results},indent=2),flush=True)

if __name__=='__main__':main()
