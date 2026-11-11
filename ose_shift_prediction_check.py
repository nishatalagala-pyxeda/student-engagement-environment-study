"""Evaluate existing student-held-out OSE forecasts on factor-shift cases."""
from pathlib import Path
import hashlib,html,json
import numpy as np
import pandas as pd
import ose_change_audit as changes
import ose_unseen_students as old

ROOT=old.ROOT;OUT=ROOT/'outputs/ose-shift-prediction-check'
PLAN={
 'question':'On held-out student observations with large environmental/other shifts, do all historical predictors improve OSE prediction beyond OSE history?',
 'predictions':'Reuse existing fixed ridge and forest held-out predictions from ose-unseen-with-history. All-versus-history-only comparisons use identical observations and algorithm. No fitting, tuning, or model selection on this subgroup analysis.',
 'primary_population':'1118 history-eligible reports138 cohort-aware IDs, same prior five student-held-out folds. Secondary all1530reports158IDs. Earlier personal OSE allowed at inference; each entire test student excluded from model training.',
 'physical_known':'Prior audit physical_forecast events: change between two nonoverlapping7day exposure windows[D-15,D-8) associated with consecutive OSE dates7–21days apart. Both periods are available before current target minus7days. Original thresholds unchanged: temperature2C,humidity10,CO2 300,noise5,PM2.5 5 in recorded units.',
 'survey_known':'For each target and factor, compare two most recent valid same-person measurements strictly before D-7. Require latest report no older than28days and reports7–21days apart. Apply original audit thresholds: environment/rating2points,PSS1,PANAS10,speed3fold plus20download/10upload. Shift membership uses no target OSE. Group environmental satisfaction separately from mood/stress/computer/network.',
 'retrospective':'Also report previous recent physical and concurrent questionnaire-shift subgroups. These shifts were not necessarily available to the forecasting model: recent sensor window[D-7,D) and same-day surveys postdate its input cutoff. They are retrospective case diagnostics, not a fair test of supplying those contemporaneous shifts as inputs.',
 'metrics':'Report-weighted MAE,RMSE,R2 and paired all-minus-history comparison. Bootstrap conservative numeric student-ID groups3000draws, fixed predictions. Positive improvement favors all predictors. Source thresholds fixed from previous audit, no selection by OSE movement.',
 'limits':'Small/dependent subgroups, many exploratory comparisons, inferred sensor IDs, sparse coverage and repeated dataset reuse. This evaluates existing historically lagged models, not a new shift-aware model or concurrent-measurement model. Cannot establish causality or absence of environmental influence.'
}

def known_survey_events(targets):
    stores={}
    for tag,path in [('satisfaction',changes.SOURCE/'cleaned_satisfaction.csv'),('computer',changes.SOURCE/'cleaned_computer.csv'),('mood',changes.SOURCE/'cleaned_mood.csv'),('stress',ROOT/'outputs/expanded-stress-increases/cleaned_stress_reports.csv')]:
        g=pd.read_csv(path,float_precision='round_trip');g.date=pd.to_datetime(g.date);stores[tag]={p:v.sort_values('date') for p,v in g.groupby('person')}
    specs=[(col,'absolute',threshold) for col,(_,threshold) in changes.SURVEYS.items()]+[('computer_DownloadSpeed','ratio',20),('computer_UploadSpeed','ratio',10)]
    records=[]
    for _,r in targets.iterrows():
        bound=r.date-pd.Timedelta(days=7)
        for factor,kind,threshold in specs:
            tag,col=factor.split('_',1);data=stores[tag].get(r.person)
            if data is None:continue
            data=data[data.date.lt(bound)&data[col].notna()].tail(2)
            if len(data)<2:continue
            a,b=data.iloc[0],data.iloc[1];gap=(b.date-a.date).days
            if not 7<=gap<=21 or (r.date-b.date).days>28:continue
            if kind=='ratio' and (a[col]<=0 or b[col]<=0):continue
            large=bool(abs(b[col]-a[col])>=threshold)
            if kind=='ratio':large=large and max(b[col]/a[col],a[col]/b[col])>=3
            records.append({'source_row':int(r.source_row),'person':r.person,'target_date':str(r.date.date()),'factor':factor,'group':'Environmental satisfaction shifts known earlier' if tag=='satisfaction' else 'Other shifts known earlier','date_before':str(a.date.date()),'date_after':str(b.date.date()),'value_before':float(a[col]),'value_after':float(b[col]),'large':large,'bound_exclusive':str(bound.date())})
    return pd.DataFrame(records)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis_plan.json').write_text(json.dumps(PLAN,indent=2))
    schema,d,cols=old.load();survey=known_survey_events(d);survey.to_csv(OUT/'known_survey_shift_audit.csv',index=False)
    events=pd.read_csv(changes.OUT/'large_change_examples.csv',float_precision='round_trip')
    sets={'Physical shifts known earlier':set(events.loc[events.group.eq('physical_forecast'),'source_row_after'])}
    for group in ['Environmental satisfaction shifts known earlier','Other shifts known earlier']:
        sets[group]=set(survey.loc[survey.group.eq(group)&survey.large,'source_row'])
    sets['Any shift known earlier']=set.union(*sets.values())
    sets['Physical shifts during recent week (retrospective)']=set(events.loc[events.group.eq('physical_recent'),'source_row_after'])
    sets['Same-day survey/network shifts (retrospective)']=set(events.loc[events.group.eq('self_report_or_network'),'source_row_after'])
    rows=[{'subset':name,'source_row':int(key)} for name,keys in sets.items() for key in sorted(keys)]
    pd.DataFrame(rows).to_csv(OUT/'subgroup_membership.csv',index=False)
    path=ROOT/'outputs/ose-unseen-with-history/predictions.csv';p=pd.read_csv(path,float_precision='round_trip',dtype={'student_group':str})
    results={}
    for protocol,pp in p.groupby('protocol'):
        results[protocol]={}
        for subset,keys in sets.items():
            sub=pp[pp.source_row.isin(keys)];scores={};comparisons={};coverage=sub[sub.model.eq('Last OSE')]
            for name,g in sub.groupby('model'):
                scores[name]=old.previous.core.regression.score(g.observed.to_numpy(),g.predicted.to_numpy())
            if len(coverage):
                for family in ['Ridge','Random forest']:
                    name='All including past OSE | '+family
                    for base in ['Past OSE only | '+family,'Last OSE']:
                        a=sub[sub.model.eq(name)].copy();b=sub[sub.model.eq(base)].copy();a.person=a.student_group;b.person=b.student_group
                        comparisons[name+' versus '+base]=old.previous.paired(a,b)
            results[protocol][subset]={'reports':len(coverage),'ids':coverage.person.nunique(),'scores':scores,'comparisons':comparisons}
    r={'plan':PLAN,'subsets':results,'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [path,changes.OUT/'large_change_examples.csv',old.SOURCE/'modeling_features.csv']}}
    (OUT/'results.json').write_text(json.dumps(r,indent=2,default=int));render(r)
    print(json.dumps({k:{s:{'n':v['reports'],'ids':v['ids'],'MAE':{name:score['mae'] for name,score in v['scores'].items()},'comparisons':v['comparisons']} for s,v in groups.items()} for k,groups in results.items()},indent=2,default=int))

def render(r):
    text='<!doctype html><meta charset="utf-8"><title>Do extra predictors help after large shifts?</title><style>body{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#253243}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f7}</style><h1>Do all predictors help specifically in large-shift cases?</h1><p>Same student-held-out predictions and unchanged factor-shift definitions. Earlier personal OSE is available at inference. Primary comparisons retain the same algorithm and records; no model was trained or selected on these subgroups.</p>'
    primary=r['subsets']['Previous eligible subset']['Any shift known earlier'];s=primary['scores'];c=primary['comparisons']['All including past OSE | Random forest versus Past OSE only | Random forest'];lo,hi=c['person_cluster_95_interval']
    text+=f'<p><strong>No demonstrated gain from the all-predictor models on large-shift cases.</strong> Among {primary["reports"]} held-out reports from {primary["ids"]} students with at least one earlier-known substantial shift, history-only forest MAE is {s["Past OSE only | Random forest"]["mae"]:.2f}, compared with {s["All including past OSE | Random forest"]["mae"]:.2f} for all predictors. The paired improvement is {c["mae_improvement"]:.2f} points, with95% interval [{lo:.2f}, {hi:.2f}]. Physical, satisfaction, and other earlier-known shift subgroups each show no demonstrated incremental improvement. This conclusion is restricted to the existing lagged models. It does not test a new model supplied with contemporaneous measurements of a newly occurring shift.</p>'
    for protocol in ['Previous eligible subset','All students']:
        text+='<h2>'+html.escape(protocol)+'</h2><table><tr><th>Subset</th><th>Reports / IDs</th><th>History-only forest MAE</th><th>All-predictor forest MAE</th><th>History-only ridge MAE</th><th>Last OSE MAE</th><th>Forest MAE gain /95% interval</th></tr>'
        for name,v in r['subsets'][protocol].items():
            s=v['scores'];c=v['comparisons']['All including past OSE | Random forest versus Past OSE only | Random forest'];lo,hi=c['person_cluster_95_interval']
            values=[s[k]['mae'] for k in ['Past OSE only | Random forest','All including past OSE | Random forest','Past OSE only | Ridge','Last OSE']]
            text+=f'<tr><td>{html.escape(name)}</td><td>{v["reports"]} / {v["ids"]}</td>'+''.join(f'<td>{x:.3f}</td>' for x in values)+f'<td>{c["mae_improvement"]:.3f} / [{lo:.3f}, {hi:.3f}]</td></tr>'
        text+='</table>'
    text+='<h2>Interpretation boundaries</h2><p>Known-earlier subsets identify substantial shifts in measurements preceding the forecasting cutoff. Retrospective rows identify cases from the prior audit, but those current-week/current-survey shifts were not actually given to the forecasting model. Failure there cannot establish that supplying current shifts would be unhelpful. No subset is selected using OSE change.</p><h2>Method and limits</h2>'+''.join('<p><strong>'+html.escape(k)+':</strong> '+html.escape(v)+'</p>' for k,v in PLAN.items())
    text+='<p><a href="results.json">Full model scores and intervals</a> · <a href="known_survey_shift_audit.csv">Known-earlier shift audit</a> · <a href="subgroup_membership.csv">Exact subgroup membership</a> · <a href="independent_checks.json">Verification</a></p>'
    (OUT/'report.html').write_text(text)

if __name__=='__main__':main()
