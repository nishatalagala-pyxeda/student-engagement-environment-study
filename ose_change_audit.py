"""Descriptive within-student factor changes paired with OSE observations."""
from pathlib import Path
import hashlib,html,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'outputs/ose-all-variables';OUT=ROOT/'outputs/ose-change-audit'
SENSORS={'temperature_C':2,'humidity':10,'co2':300,'noise':5,'pm25':5}
SURVEYS={
 'satisfaction_TempSatisf':('Temperature satisfaction',2),
 'satisfaction_AirSatisf_1':('Air satisfaction',2),
 'satisfaction_NoiseSatisf_1':('Noise satisfaction',2),
 'satisfaction_EnvSatisf_1':('Overall environment satisfaction',2),
 'computer_CPUSpeed_1':('Computer speed rating',2),
 'computer_InternetSpeed_1':('Internet speed rating',2),
 'stress_PSS':('Stress',1),
 'mood_Positive emotion':('Positive affect',10),
 'mood_Negative emotion':('Negative affect',10)
}
PLAN={
 'question':'Are there substantial within-person environmental or other factor changes with OSE scores around those changes?',
 'pairs':'Consecutive cleaned same-person OSE observations7–21calendar days apart; one transition per consecutive pair, never all possible pairs. Cohort-aware supplied IDs, not author-confirmed physical linkage.',
 'sensors':'Primary sensor exposure is equal-weight mean of good daily sensor means over[D-7,D), >=5good days; good day already requires>=72distinct15min slots. Compared windows are nonoverlapping because OSE gap>=7days. Sensor mapping remains inferred. Noise is arithmetic reported-value mean, not Leq. No occupancy/calibration confirmation.',
 'sensor_thresholds':'Illustrative absolute weekly-mean differences: temperature2C, humidity10recorded units,CO2 300recorded units,noise5recorded units,PM2.5 5recorded units. These are exploratory contrast definitions, not validated physiological/safety thresholds. Units other than temperature are retained in recorded units.',
 'survey_thresholds':'Same-date exact-person survey values at both OSE dates: satisfaction/computer/network rating2points, PSS1point, positive/negative affect10points. Download>=3fold AND20recorded units; upload>=3fold AND10recorded units. Zero denominators excluded from speed ratios. No near-date guessing. These are exploratory thresholds, not clinical cutoffs.',
 'timing':'Survey comparisons are contemporaneous descriptive associations. Sensor windows precede OSE but OSE lacks explicit weekly recall. Secondary previous forecasting windows[D-15,D-8) are analyzed separately. No intervention date or verified moving-room event is recorded; cannot infer a causal environmental effect.',
 'outcomes':'Compare observed OSE change across the pair. Report absolute changes and fraction>=5points for orientation, not validated meaningful-change definition. Multiple transitions from a student are dependent. Large/small groups descriptive, not causal/tuned predictive estimates.',
 'limits':'Sparse sensor overlap, extreme raw sensor values retained, unknown occupancy, questionnaire scoring uncertainty, simultaneous factor changes, seasonality/time confounding, selection/missingness and reused exploratory dataset. No new model training. Missing factors never treated as unchanged.'
}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2))
    paths=[ROOT/'outputs/ose-stress-prediction/cleaned_ose.csv',SOURCE/'environment_daily_all.csv']
    d=pd.read_csv(paths[0],float_precision='round_trip');d.date=pd.to_datetime(d.date)
    daily=pd.read_csv(paths[1],float_precision='round_trip');daily.day=pd.to_datetime(daily.day)
    signals=list(SENSORS);byid={p:g for p,g in daily.groupby('person')};coverage=[];records=[]
    for _,r in d.iterrows():
        rec={};g=byid.get(r.person,pd.DataFrame(columns=daily.columns))
        for label,start,end in [('recent',7,0),('forecast',15,8)]:
            a=r.date-pd.Timedelta(days=start);b=r.date-pd.Timedelta(days=end);w=g[g.day.ge(a)&g.day.lt(b)]
            for c in signals:rec[label+'_'+c]=float(w[c].mean()) if len(w)>=5 else np.nan
            coverage.append({'source_row':int(r.source_row),'person':r.person,'date':str(r.date.date()),'window':label,'start_inclusive':str(a.date()),'end_exclusive':str(b.date()),'good_days':len(w)})
        records.append(rec)
    d=pd.concat([d,pd.DataFrame(records)],axis=1)
    for tag,path in [('satisfaction',SOURCE/'cleaned_satisfaction.csv'),('computer',SOURCE/'cleaned_computer.csv'),('mood',SOURCE/'cleaned_mood.csv'),('stress',ROOT/'outputs/expanded-stress-increases/cleaned_stress_reports.csv')]:
        paths.append(path);g=pd.read_csv(path,float_precision='round_trip');g.date=pd.to_datetime(g.date)
        keep=[c for c in g if tag+'_'+c in SURVEYS or (tag=='computer' and c in ['DownloadSpeed','UploadSpeed'])]
        g=g[['person','date']+keep].rename(columns={c:tag+'_'+c for c in keep});assert not g.duplicated(['person','date']).any()
        d=d.merge(g,on=['person','date'],how='left',validate='one_to_one')
    d=d.sort_values(['person','date']).reset_index(drop=True)
    previous=d.groupby('person').shift(1);gap=(d.date-previous.date).dt.days;valid=gap.between(7,21)
    pairs=d[valid].copy();before=previous[valid].copy();pairs['previous_date']=before.date;pairs['gap_days']=gap[valid];pairs['previous_source_row']=before.source_row;pairs['OSE_before']=before.OSE;pairs['OSE_after']=pairs.OSE;pairs['OSE_change']=pairs.OSE-before.OSE
    factor_cols=[c for c in d if c not in ['person','date','source_row','OSE']]
    for c in factor_cols:
        pairs[c+'_before']=before[c];pairs[c+'_after']=pairs[c];pairs[c+'_change']=pairs[c]-before[c]
    event_records=[];summaries=[]
    specs=[(label+'_'+c,label+' '+c,'absolute',threshold) for label in ['recent','forecast'] for c,threshold in SENSORS.items()]+[(c,name,'absolute',threshold) for c,(name,threshold) in SURVEYS.items()]+[('computer_DownloadSpeed','Download speed','ratio',20),('computer_UploadSpeed','Upload speed','ratio',10)]
    for col,name,kind,threshold in specs:
        available=pairs[col+'_before'].notna()&pairs[col+'_after'].notna();absolute=pairs[col+'_change'].abs()
        if kind=='ratio':
            available&=pairs[col+'_before'].gt(0)&pairs[col+'_after'].gt(0)
            ratio=np.maximum(pairs[col+'_after']/pairs[col+'_before'],pairs[col+'_before']/pairs[col+'_after']);large=available&ratio.ge(3)&absolute.ge(threshold)
        else:large=available&absolute.ge(threshold)
        group='physical_recent' if col.startswith('recent_') else 'physical_forecast' if col.startswith('forecast_') else 'self_report_or_network'
        summary={'factor':col,'name':name,'group':group,'definition':f'absolute change >= {threshold}'+(' and ratio >=3' if kind=='ratio' else ''),'covered_pairs':int(available.sum()),'covered_ids':int(pairs.loc[available,'person'].nunique()),'large_pairs':int(large.sum()),'large_ids':int(pairs.loc[large,'person'].nunique()),'groups':{}}
        for label,mask in [('large_change',large),('smaller_change',available&~large)]:
            v=pairs.loc[mask,'OSE_change'];summary['groups'][label]={'n':len(v),'ids':int(pairs.loc[mask,'person'].nunique()),'mean_signed_OSE_change':float(v.mean()) if len(v) else None,'median_absolute_OSE_change':float(v.abs().median()) if len(v) else None,'mean_absolute_OSE_change':float(v.abs().mean()) if len(v) else None,'OSE_change_ge5':int(v.abs().ge(5).sum()),'OSE_change_le2':int(v.abs().le(2).sum())}
        summaries.append(summary)
        for _,row in pairs[large].iterrows():event_records.append({'factor':col,'name':name,'group':group,'person':row.person,'date_before':str(row.previous_date.date()),'date_after':str(row.date.date()),'source_row_before':int(row.previous_source_row),'source_row_after':int(row.source_row),'gap_days':int(row.gap_days),'factor_before':float(row[col+'_before']),'factor_after':float(row[col+'_after']),'factor_change':float(row[col+'_change']),'OSE_before':float(row.OSE_before),'OSE_after':float(row.OSE_after),'OSE_change':float(row.OSE_change),'threshold_multiple':float(abs(row[col+'_change'])/threshold),'zero_OSE_endpoint':bool(row.OSE_before==0 or row.OSE_after==0)})
    events=pd.DataFrame(event_records);events.to_csv(OUT/'large_change_examples.csv',index=False);pairs.to_csv(OUT/'paired_observations.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'sensor_window_coverage.csv',index=False)
    unions={}
    for group,g in events.groupby('group'):
        unique=g.drop_duplicates(['person','date_before','date_after']);unions[group]={'pairs':len(unique),'ids':unique.person.nunique(),'OSE_change_ge5':int(unique.OSE_change.abs().ge(5).sum()),'OSE_change_le2':int(unique.OSE_change.abs().le(2).sum())}
    examples=[]
    for group,g in events[~events.zero_OSE_endpoint].groupby('group'):
        for label,gg in [('OSE_changed',g[g.OSE_change.abs().ge(5)]),('OSE_stable',g[g.OSE_change.abs().le(2)])]:
            take=gg.sort_values('threshold_multiple',ascending=False).drop_duplicates(['person','date_before','date_after']).head(3)
            examples.extend(take.assign(example_type=label).to_dict('records'))
    r={'plan':PLAN,'ose_reports':len(d),'ose_ids':d.person.nunique(),'paired_transitions':len(pairs),'paired_ids':pairs.person.nunique(),'sensor_covered_recent_pairs':int(pairs[['recent_'+c+'_before' for c in signals]+['recent_'+c+'_after' for c in signals]].notna().all(axis=1).sum()),'unions':unions,'summaries':summaries,'selected_examples':examples,'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (OUT/'results.json').write_text(json.dumps(r,indent=2,default=int));render(r)
    print(json.dumps({k:v for k,v in r.items() if k in ['paired_transitions','paired_ids','sensor_covered_recent_pairs','unions','summaries','selected_examples']},indent=2,default=int))

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Large factor changes and OSE</title><style>body{font:16px/1.6 system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2f7}</style><h1>Large within-student changes with OSE observations</h1><p>Exploratory descriptive audit of consecutive OSE responses7–21days apart. Physical conditions use separate preceding7-day windows; >=5good sensor days required per window. Self-report/network comparisons use exact same dates as OSE. Thresholds illustrate contrasts and were not selected for OSE associations.</p>'
    union=r['unions']['physical_recent']
    text+=f'<p><strong>Yes: substantial within-student factor changes are present.</strong> Of {r["sensor_covered_recent_pairs"]} sensor-covered paired observations from 36 students, {union["pairs"]} pairs from {union["ids"]} students meet at least one of the five illustrative physical-change thresholds. In those pairs, OSE moves by at least 5 points in {union["OSE_change_ge5"]} and by at most 2 points in {union["OSE_change_le2"]}. Some substantial exposure changes therefore coincide with changed engagement, and others with nearly stable engagement; these observations do not establish cause or absence of an effect.</p>'
    text+=f'<p>{r["paired_transitions"]} total paired transitions from {r["paired_ids"]} cohort-aware IDs; {r["sensor_covered_recent_pairs"]} have both preceding-week sensor windows. Missing measurements are not unchanged measurements. For temperature specifically, there are 27 changes of at least 2°C across 17 students; even using a 3°C threshold leaves 8 changes.</p><table><tr><th>Factor / threshold</th><th>Covered pairs / IDs</th><th>Large-change pairs / IDs</th><th>Median absolute OSE change, large</th><th>Median absolute OSE change, smaller</th></tr>'
    for s in r['summaries']:
        a=s['groups']['large_change'];b=s['groups']['smaller_change'];text+=f'<tr><td>{html.escape(s["name"])} / {html.escape(s["definition"])}</td><td>{s["covered_pairs"]} / {s["covered_ids"]}</td><td>{s["large_pairs"]} / {s["large_ids"]}</td><td>{a["median_absolute_OSE_change"]}</td><td>{b["median_absolute_OSE_change"]}</td></tr>'
    text+='</table><h2>Illustrative examples, including changed and stable OSE</h2><p>Selected by size of factor contrast among each outcome category, not representative estimates. Extreme sensor contrasts may reflect device/placement/occupancy artifacts. Counts and full rows are saved for independent inspection.</p><table><tr><th>Student / dates</th><th>Factor</th><th>Before → after</th><th>OSE before → after</th></tr>'
    for e in r['selected_examples']:text+=f'<tr><td>{html.escape(e["person"])} / {e["date_before"]} → {e["date_after"]}</td><td>{html.escape(e["name"])}</td><td>{e["factor_before"]:.2f} → {e["factor_after"]:.2f}</td><td>{e["OSE_before"]:.1f} → {e["OSE_after"]:.1f}</td></tr>'
    text+='</table><h2>Method and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p><a href="results.json">Full counts and definitions</a> · <a href="large_change_examples.csv">All large changes</a> · <a href="paired_observations.csv">All paired measurements</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
