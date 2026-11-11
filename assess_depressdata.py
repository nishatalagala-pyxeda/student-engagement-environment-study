"""Read-only DEPRESSData inventory, join feasibility and target quality assessment. No model fitting."""
from pathlib import Path
import json,re,subprocess,hashlib,html,zipfile,xml.etree.ElementTree as E,io
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Users/nishatalagala/Downloads/DEPRESSData')
OUT=ROOT/'outputs/depressdata-assessment'
OUT.mkdir(parents=True,exist_ok=True)

def canonical(value):
    if pd.isna(value):return None
    text=re.sub(r'\s+',' ',str(value).strip().replace('_',' '))
    text=re.sub(r'^Sping\s+','Spring ',text,flags=re.I)
    match=re.fullmatch(r'(fall|spring|summer)\s*(\d+)',text,re.I)
    if match:return match[1].title()+' '+str(int(match[2]))
    return text

def read_tables():
    data={};audit={}
    for path in sorted(SOURCE.rglob('*.csv')):
        d=pd.read_csv(path,low_memory=False);name=path.stem;data[name]=d
        datecol=next(c for c in d if 'date' in c.lower() or 'timestamp' in c.lower());t=pd.to_datetime(d[datecol],errors='coerce',format='mixed')
        numeric=d.select_dtypes('number')
        audit[name]={'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(d),'columns':list(d),'fully_blank_rows':int(d.isna().all(axis=1).sum()),'missing_by_column':d.isna().sum().to_dict(),'unique_raw_ids':int(d.ID.nunique()),'unique_normalized_ids':int(d.ID.map(canonical).nunique()),'date_column':datecol,'first':str(t.min()),'last':str(t.max()),'unique_dates':int(t.dt.normalize().nunique()),'unparseable_nonempty_dates':d.loc[d[datecol].notna()&t.isna(),datecol].astype(str).tolist(),'numeric_summary':numeric.describe().to_dict()}
        d['_person']=d.ID.map(canonical);d['_time']=t;d['_date']=t.dt.normalize();d['_source_row']=d.index+2
        usable=d[d._person.notna()&d._time.notna()].copy()
        audit[name]['usable_id_date_rows']=len(usable);audit[name]['duplicate_id_timestamp_rows']=int(usable.duplicated(['_person','_time'],False).sum())
    workbook=next(SOURCE.rglob('*.xlsx'));demo=pd.read_excel(workbook);data['demographics']=demo
    demo['_person']=demo.ID.map(canonical)
    audit['demographics']={'file':str(workbook),'sheet':'Demograph&SES','rows':len(demo),'columns':[c for c in demo if not c.startswith('_')],'unique_normalized_ids':int(demo._person.nunique()),'missing_by_column':demo.isna().sum().to_dict(),'duplicate_ids':demo.loc[demo._person.duplicated(False),'_person'].dropna().tolist()}
    return data,audit

def labels_clean(d,targets):
    used=d[d._person.notna()&d._date.notna()].copy()
    conflict=used.groupby(['_person','_date'])[targets].transform('nunique').gt(1).any(axis=1)
    nonconflict=used[~conflict];clean=nonconflict.drop_duplicates(['_person','_date']).copy()
    return clean,{'id_date_rows':len(used),'conflicting_same_person_date_rows':int(conflict.sum()),'redundant_rows_collapsed':len(nonconflict)-len(clean),'clean_rows':len(clean),'clean_people':int(clean._person.nunique()),'nonmissing_target_counts':clean[targets].notna().sum().to_dict()}

def archive_inventory():
    records=[];fit_members=[];face_ids=set();face_dates=[]
    for p in sorted(SOURCE.rglob('*.rar')):
        proc=subprocess.run(['/usr/bin/bsdtar','-tf',str(p)],capture_output=True,text=True)
        members=proc.stdout.splitlines();csv=[n for n in members if n.lower().endswith('.csv')]
        records.append({'file':str(p),'bytes':p.stat().st_size,'listing_returncode':proc.returncode,'member_count':len(members),'csv_count':len(csv),'listing_warning':proc.stderr.strip()})
        if p.name=='Fitbit.rar':fit_members=csv;fitpath=p
        else:
            for member in csv:
                parts=member.split('/')
                if len(parts)>=3:face_ids.add(canonical(parts[-2]))
                match=re.search(r'[-_](\d{8})',Path(member).name)
                if match:
                    t=pd.to_datetime(match[1],format='%d%m%Y',errors='coerce')
                    if pd.notna(t):face_dates.append(t)
    fit_ids=set();types={}
    for member in fit_members:
        parts=member.split('/')
        if len(parts)>2:fit_ids.add(canonical(parts[1]))
        typ=re.sub(r'\d.*','',Path(member).stem);types[typ]=types.get(typ,0)+1
    # Sample only a few short CSV headers, never expand the large archives in full.
    samples={}
    for typ in types:
        member=next(n for n in fit_members if re.sub(r'\d.*','',Path(n).stem)==typ)
        proc=subprocess.Popen(['/usr/bin/bsdtar','-xOf',str(fitpath),member],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        raw=proc.stdout.read(4096);proc.terminate();proc.communicate()
        lines=raw.decode('utf-8',errors='replace').splitlines()
        samples[typ]={'member':member,'header':lines[0] if lines else '', 'sample_rows_read':max(0,len(lines)-1)}
    return {'archives':records,'fitbit_csv_files':len(fit_members),'fitbit_people':len(fit_ids),'fitbit_ids':sorted(fit_ids),'fitbit_file_types':types,'fitbit_header_samples':samples,'facial_csv_files':sum(v['csv_count'] for v in records if Path(v['file']).name!='Fitbit.rar'),'facial_people':len(face_ids),'facial_ids':sorted(face_ids),'facial_first_filename_date':str(min(face_dates)) if face_dates else None,'facial_last_filename_date':str(max(face_dates)) if face_dates else None,'warning':'Archive metadata and sampled Fitbit headers only; raw wearable/facial values and completeness are not yet validated. Facial features are tracking/gaze/landmark/action-unit data, not independently labeled emotions.'}

def main():
    data,audit=read_tables();print('Table inventory complete',flush=True)
    universe=set().union(*(set(d._person.dropna()) for name,d in data.items() if name!='Bedroom_IEQ'))
    env=data['Bedroom_IEQ'].copy();mapping={};mapaudit=[]
    for rawid,g in env.groupby('ID'):
        num=str(rawid)[1:];poss=[f'Fall {num}'] if str(rawid).startswith('F') else [f'Summer {num}',f'Spring {num}'];candidates=[v for v in poss if v in universe]
        proposed=candidates[0] if len(candidates)==1 else None;mapping[rawid]=proposed
        mapaudit.append({'sensor_id':rawid,'proposed_person':proposed,'candidates':candidates,'status':'Unique lexical candidate; inferred, not author-confirmed' if proposed else 'Unresolved; do not force a match','rows':len(g),'first':str(g._time.min()),'last':str(g._time.max())})
    env['_person']=env.ID.map(mapping)
    # Infer the repeated New York fall-back hour from original per-person recording order.
    # All 72 repeated wall-clock rows are 2020-11-01 01:00..01:45; they are not bad duplicate readings.
    utc_parts=[]
    for _,g in env.groupby('ID',sort=False):
        localized=pd.DatetimeIndex(g._time).tz_localize('America/New_York',ambiguous='infer',nonexistent='NaT').tz_convert('UTC')
        assert localized.notna().all(), 'Nonexistent/unresolved local sensor timestamp'
        utc_parts.append(pd.Series(localized,index=g.index))
    env['_utc']=pd.concat(utc_parts).sort_index()
    valuecols=[c for c in audit['Bedroom_IEQ']['columns'] if c not in ['ID',audit['Bedroom_IEQ']['date_column']]]
    counts=env.groupby(['ID','_utc'])[valuecols].nunique().max(axis=1)
    badkeys=set(counts[counts.gt(1)].index)
    env['_conflict']=[(i,t) in badkeys for i,t in zip(env.ID,env._utc)]
    envclean=env[~env._conflict].drop_duplicates(['ID','_utc'])
    usable_env=envclean[envclean._person.notna()]
    daily=usable_env.groupby(['_person','_date'])._time.nunique().rename('slots').reset_index()
    daily.to_csv(OUT/'environment_daily_coverage.csv',index=False)
    pd.DataFrame(mapaudit).to_json(OUT/'sensor_id_mapping.json',orient='records',indent=2)
    rows=[];labels={};pairs={}
    sets={'weekly':('PANAS_PSS',['PSS','Positive emotion','Negative emotion']),'monthly':('CES-D_STAI',['CESD','STAI_st']),'comfort':('General_Environmental_Satisfaction_Survey',['ThermalSens','EnvSatisf_1']),'self_efficacy':('OSE_Computer_Internert',['OSE'])}
    envby={person:g.set_index('_date').slots for person,g in daily.groupby('_person')}
    for kind,(name,targets) in sets.items():
        clean,info=labels_clean(data[name],targets);labels[kind]=info
        for _,row in clean.iterrows():
            days=pd.date_range(row._date-pd.Timedelta(days=7),periods=7,freq='D');coverage=envby.get(row._person,pd.Series(dtype=float)).reindex(days,fill_value=0)
            rec={'table':kind,'source_row':int(row._source_row),'person':row._person,'survey_date':str(row._date.date()),'past_7d_slots':int(coverage.sum()),'past_7d_days_any':int(coverage.gt(0).sum()),'past_7d_days_72plus_slots':int(coverage.ge(72).sum()),'has_sensor_identity':row._person in set(mapping.values()),**{c:row[c] for c in targets}}
            rec['at_least_5_days_72plus_slots']=rec['past_7d_days_72plus_slots']>=5;rows.append(rec)
        selected=pd.DataFrame([v for v in rows if v['table']==kind]);qualified=selected[selected.at_least_5_days_72plus_slots]
        pairs[kind]={'clean_label_rows':len(clean),'clean_label_people':int(clean._person.nunique()),'labels_with_any_preceding_week_sensor_data':int(selected.past_7d_slots.gt(0).sum()),'people_with_any_preceding_week_sensor_data':int(selected.loc[selected.past_7d_slots.gt(0),'person'].nunique()),'labels_with_5_days_each_72plus_slots':len(qualified),'people_with_5_days_each_72plus_slots':int(qualified.person.nunique()),'qualified_dates':int(qualified.survey_date.nunique()),'qualified_nonmissing_targets':qualified[targets].notna().sum().to_dict()}
    windows=pd.DataFrame(rows);windows.to_csv(OUT/'survey_environment_overlap.csv',index=False)
    weekly=data['PANAS_PSS'];weekly_valid=weekly[weekly._person.notna()&weekly._date.notna()]
    score_issues={'weekly_zero_positive_emotion':int(weekly_valid['Positive emotion'].eq(0).sum()),'weekly_zero_negative_emotion':int(weekly_valid['Negative emotion'].eq(0).sum()),'weekly_PANAS_below10_rows':int(weekly_valid[['Positive emotion','Negative emotion']].lt(10).any(axis=1).sum()),'weekly_missing_PSS_with_id_date':int(weekly_valid.PSS.isna().sum()),'PSS_observed_values':sorted(weekly_valid.PSS.dropna().unique().tolist()),'CESD_observed_range':[int(data['CES-D_STAI'].CESD.min()),int(data['CES-D_STAI'].CESD.max())],'notes':['Questionnaire image shows four-item PSS about the previous week; provided PSS ranges0..4 in0.25 increments, consistent with an item average, not a conventional summed score. Raw PSS items absent, so scoring/reverse coding cannot be independently checked.','PANAS image shows20 affect items with five response options; only positive/negative aggregate scores are provided. Zero/sub-10 values need coding clarification; do not silently treat them as low mood or zero-fill missing labels.','CES-D image has10 questions; do not apply20-item thresholds. Only aggregate CESD is supplied, so its item scoring cannot be reconstructed.','STAI has20 raw item columns plus total; item wording and reverse-scoring should be verified before modeling, and those items must not predict their own total.']}
    sensor={'raw_ids':int(env.ID.nunique()),'uniquely_inferred_ids':sum(v is not None for v in mapping.values()),'unresolved_ids':[k for k,v in mapping.items() if v is None],'raw_rows':len(env),'same_id_timestamp_duplicate_rows':int(env.duplicated(['ID','_time'],False).sum()),'conflicting_duplicate_keys':len(badkeys),'conflicting_duplicate_rows':int(env._conflict.sum()),'deduplicated_unconflicted_rows':len(envclean),'linked_rows':len(usable_env),'sensor_days':len(daily),'validity_ranges':env[valuecols].agg(['min','max']).to_dict(),'mapping_note':'Participant-level matching is much more direct than the prior classroom mapping, but short-code expansion is still inferred; F135 is unresolved and excluded. S codes match Summer/Spring only when exactly one candidate exists in supplied survey/demographic IDs. No numeric-only or date-based forced mapping.','coverage_note':'Previous seven complete calendar days, ending before the survey date (no survey time supplied). Strong-coverage feasibility means at least5 days each with at least72 distinct15-minute wall-clock slots. This is a declared assessment threshold, not a validated missing-data rule. America/New_York timestamps are naive; DST ambiguity and participant-local Fitbit times need resolution before fine-grained alignment.'}
    stai=data['CES-D_STAI'];items=[f'STAI_St_{i}' for i in range(1,21)];recoded=stai[items].copy()
    reverse_items=[1,2,5,8,10,11,15,16,19,20]
    for i in reverse_items:recoded[f'STAI_St_{i}']=5-recoded[f'STAI_St_{i}']
    score_issues['STAI_total_equals_stored_item_sum_rows']=int(stai[items].sum(axis=1).eq(stai.STAI_st).sum())
    score_issues['STAI_total_equals_reverse_positive_wording_sum_rows']=int(recoded.sum(axis=1).eq(stai.STAI_st).sum())
    score_issues['STAI_reverse_items_from_questionnaire_wording']=reverse_items
    score_issues['notes'][-1]='STAI_st equals the simple sum of its20 stored item columns in all558 rows. This is appropriate only if the positive-worded items were already reverse-coded. The supplied documentation does not establish that; clarify coding before treating this total as state anxiety. None of these items may predict their own total.'
    score_issues['qualified_weekly_zero_or_sub10_PANAS_rows']=int(windows.loc[windows.table.eq('weekly')&windows.at_least_5_days_72plus_slots,['Positive emotion','Negative emotion']].lt(10).any(axis=1).sum())
    sensor['coverage_note']='Previous seven complete calendar days, ending before the survey date (no survey time supplied). Feasibility threshold: at least5 days each with at least72 distinct15-minute wall-clock slots. This is a declared assessment threshold, not a validated missing-data rule. Source New York local time is localized using original per-person order, including the repeated fall-back hour. Coverage counts distinct wall-clock slots conservatively; participant-local Fitbit time zones still need alignment.'
    sensor['daylight_saving_note']='All72 repeated local timestamp rows are2020-11-01 01:00..01:45 in nine sensors. They resolve to distinct UTC instants under New York fall-back inference. They are retained as valid readings, not discarded as conflicting measurements.'
    sensor['unique_real_timestamp_rows']=sensor.pop('deduplicated_unconflicted_rows')
    print('Sensor/label overlap complete',flush=True);archives=archive_inventory();print('Archive inventories complete',flush=True)
    archives['facial_date_parsing_note']='Filename dates are only provisionally parsed DDMMYYYY following the README example. The resulting range extends outside the other supplied tables; mixed conventions or filename errors must be checked before aligning facial features. Do not use the parsed extrema as verified collection dates.'
    extra={'reported_grades_file_present':bool(list(SOURCE.rglob('Grades.csv'))),'actual_sensor_measurement_missingness':'All raw IEQ measurement fields nonmissing, but range/plausibility and wear/occupancy require additional validation. Bedroom sensor readings do not prove participant presence.','demographics_people_overlap':{k:len(set(d._person.dropna())&set(data['demographics']._person.dropna())) for k,d in data.items() if k!='demographics'},'id_typo_rows':{k:int(d.ID.fillna('').astype(str).str.match(r'(?i)^sping\s').sum()) for k,d in data.items()},'models_trained':False}
    extra['demographics_people_overlap']['Bedroom_IEQ']=len(set(usable_env._person)&set(data['demographics']._person.dropna()))
    extra['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result={'source':str(SOURCE),'tables':audit,'labels':labels,'sensor':sensor,'overlap':pairs,'score_issues':score_issues,'archives':archives,'additional_checks':extra}
    (OUT/'assessment.json').write_text(json.dumps(result,indent=2,default=str));render(result)
    print(json.dumps({'labels':labels,'sensor':{k:sensor[k] for k in ['raw_ids','uniquely_inferred_ids','unresolved_ids','raw_rows']},'overlap':pairs,'archives':{k:archives[k] for k in ['fitbit_people','fitbit_csv_files','facial_people','facial_csv_files']},'score_issues':score_issues},indent=2),flush=True)

def render(r):
    text='''<!doctype html><html><head><meta charset="utf-8"><title>DEPRESSData feasibility assessment</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}h2{margin-top:32px}</style></head><body><h1>DEPRESSData: suitability for prediction</h1><p>Read-only assessment, November10,2026. No models trained.</p><p>The supplied README describes “Dataset on Emotions, Performance, Responses, Environment, and Satisfaction of Students in COVID-19 E-learning”. Bedroom AWAIR sensors, repeated weekly/monthly questionnaires, daily routines, demographics, Fitbit and facial tracking features are present. Files are spread across download batches; inventory treats them as one dataset without modifying sources.</p><h2>Available tables</h2><table><tr><th>Source</th><th>Rows</th><th>Distinct IDs after normalization</th><th>Coverage</th></tr>'''
    for name,a in r['tables'].items():text+=f'<tr><td>{html.escape(name)}</td><td>{a["rows"]:,}</td><td>{a["unique_normalized_ids"]}</td><td>{html.escape(a.get("first", "Static"))} – {html.escape(a.get("last", ""))}</td></tr>'
    text+='</table><h2>Environmental linkage</h2><p>'+html.escape(r['sensor']['mapping_note'])+'</p><p>'+html.escape(r['sensor']['coverage_note'])+'</p><table><tr><th>Target table</th><th>Clean label rows</th><th>Any prior-week sensor data</th><th>At least5 days with72+ slots</th><th>People at that coverage</th></tr>'
    for name,v in r['overlap'].items():text+=f'<tr><td>{name}</td><td>{v["clean_label_rows"]}</td><td>{v["labels_with_any_preceding_week_sensor_data"]}</td><td>{v["labels_with_5_days_each_72plus_slots"]}</td><td>{v["people_with_5_days_each_72plus_slots"]}</td></tr>'
    text+='</table><h2>Best first experiment</h2><p>Test weekly perceived-stress (PSS) regression using the preceding week of personal bedroom temperature, humidity, CO₂, VOCs, PM2.5, light and noise. Compare environment alone, demographics/time, and their combination, with train-only preprocessing. Weekly positive/negative affect regression is another close match once score coding is clarified. Monthly CESD/STAI offer fewer matched labels and different target cadence. These are questionnaire scores, not diagnoses. Self-efficacy is also available; OSE is not the previous dataset’s engagement scale. Thermal sensation/environmental satisfaction are additional outcomes.</p><p>Use held-out participants and a forward-time test with a seven-day gap to avoid overlapping feature windows, and train-fold mean/median baselines. A past-person-score baseline is appropriate for a separate known-student forecasting test if earlier labels exist. Do not train on target items, concurrent alternate mental-health scores, participant IDs or future records. Exposure-week reconstruction and genuine next-week forecasting are different tasks: the former uses the week the questionnaire describes, the latter must use an earlier week.</p><h2>Issues to settle before modeling</h2><ul>'
    for note in r['score_issues']['notes']:text+='<li>'+html.escape(note)+'</li>'
    text+='<li>Duplicate surveys, blank rows, an invalid date, ID spelling variants and unresolved sensor F135 require explicit cleaning. Only one unambiguous ID candidate is retained in the overlap audit.</li><li>Not all survey participants have sensors or demographics. Several weeks of measurements from one person do not create independent new participants.</li><li>Bedroom exposure does not establish where a student was during learning. General environmental satisfaction and other same-survey answers should be separate assisted estimates, not inputs to an environment-only model.</li></ul>'
    a=r['archives'];text+=f'''<h2>Archived signals and completeness</h2><p>Fitbit archive: {a['fitbit_people']} participant folders, {a['fitbit_csv_files']:,} CSV files. Facial archives: {a['facial_people']} participant folders, {a['facial_csv_files']:,} CSV files across18 archives. Only member listings and short Fitbit headers were inspected; no full33GB extraction or raw-signal validation. Facial files contain tracking/gaze/landmark/action-unit features, not independent emotion ground truth. Timing/quality should be validated before adding these modalities. Grades.csv is mentioned in the README but is absent from the supplied folder.</p><p><a href="assessment.json">Full inventory, score issues and coverage counts</a>; <a href="survey_environment_overlap.csv">Per-survey exposure coverage</a>; <a href="sensor_id_mapping.json">Proposed ID mapping and unresolved cases</a>; <a href="environment_daily_coverage.csv">Daily sensor coverage</a>. Questionnaire images15–19 and extracted documentation text are saved for local inspection. Participant-level audit files are research intermediates. Assessment reproducible with assess_depressdata.py.</p></body></html>'''
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
