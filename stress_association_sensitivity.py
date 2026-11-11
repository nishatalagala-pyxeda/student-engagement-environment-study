"""Diagnostics prompted by the exploratory within-person association; not new confirmatory tests."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import stress_environment_hypothesis as main
OUT=main.OUT

def inputs(d,history_adjusted=True):
    d=d.copy();d['calendar_week']=d.date.dt.to_period('W-SUN').astype(str)
    names=list(main.PLAN['exposure_units']);z=d[names].to_numpy(float);z[:,1]/=10;z[:,2]/=100
    for j in [3,4,5]:z[:,j]=np.log2(1+z[:,j])
    z[:,6]/=5
    base=pd.get_dummies(d[['person','calendar_week']],drop_first=False,dtype=float).to_numpy()
    if history_adjusted:
        h=d[['prior_stress','past_mean_stress','days_since_prior_stress']].to_numpy(float);sd=h.std(axis=0);h=(h-h.mean(axis=0))/np.where(sd>0,sd,1);base=np.column_stack([base,h])
    return base,z,d.PSS.to_numpy(float),d.person.to_numpy()


def run():
    d=pd.read_csv(OUT/'within_person_cohort.csv');d['date']=pd.to_datetime(d.survey_date);rows=[]
    for person in sorted(d.person.unique()):
        part=d[d.person.ne(person)].copy();r=main.partial_fit(*inputs(part),wild_draws=0)
        rows.append({'omitted_person':person,'n':len(part),'VOC_coefficient':float(r['beta'][3]),'VOC_ci_low':float(r['low'][3]),'VOC_ci_high':float(r['high'][3]),'joint_F_approx_p':r['joint_F_approx_p']})
    lopo=pd.DataFrame(rows);lopo.to_csv(OUT/'leave_one_person_out_association_diagnostics.csv',index=False)
    # First differences use observed consecutive nonoverlapping weeks; person fixed levels cancel.
    differences=[]
    for person,g in d.groupby('person'):
        g=g.sort_values('date');_,z,y,_=inputs(g)
        for i in range(1,len(g)):
            gap=(g.iloc[i].date-g.iloc[i-1].date).days
            if 7<=gap<=14:
                differences.append({'person':person,'date':g.iloc[i].date,'previous_date':g.iloc[i-1].date,'delta_y':y[i]-y[i-1],'delta_prior_stress':g.iloc[i].prior_stress-g.iloc[i-1].prior_stress,'delta_past_mean':g.iloc[i].past_mean_stress-g.iloc[i-1].past_mean_stress,'delta_prior_age':g.iloc[i].days_since_prior_stress-g.iloc[i-1].days_since_prior_stress,**{f'z{j}':z[i,j]-z[i-1,j] for j in range(7)}})
    diff=pd.DataFrame(differences);diff['calendar_week']=diff.date.dt.to_period('W-SUN').astype(str)
    # Difference the calendar indicators themselves: intervals vary from7 to14 days.
    # Destination-week indicators alone would not represent the same fixed-effect adjustment.
    diff['previous_calendar_week']=pd.to_datetime(diff.previous_date).dt.to_period('W-SUN').astype(str)
    periods=sorted(set(diff.calendar_week)|set(diff.previous_calendar_week))
    base=(pd.get_dummies(diff.calendar_week,dtype=float).reindex(columns=periods,fill_value=0)-pd.get_dummies(diff.previous_calendar_week,dtype=float).reindex(columns=periods,fill_value=0)).to_numpy()
    h=diff[['delta_prior_stress','delta_past_mean','delta_prior_age']].to_numpy(float);sd=h.std(axis=0);h=(h-h.mean(axis=0))/np.where(sd>0,sd,1);base=np.column_stack([base,h])
    r=main.partial_fit(base,diff[[f'z{j}' for j in range(7)]].to_numpy(),diff.delta_y.to_numpy(),diff.person.to_numpy(),wild_draws=1999)
    diff.to_csv(OUT/'first_difference_cohort.csv',index=False)
    coefficients=[{'exposure':name,'coefficient':float(r['beta'][j]),'ci_low':float(r['low'][j]),'ci_high':float(r['high'][j]),'q_BH_7_tests':float(r['q'][j])} for j,name in enumerate(main.PLAN['exposure_units'])]
    first={k:v for k,v in r.items() if k not in ['projection','residualized_exposure','residuals','covariance','beta','se','low','high','p','q']};first['coefficients']=coefficients
    results={'status':'Exploratory post-result robustness diagnostics. Do not count repeated omission p-values as independent confirmations.','leave_one_person_out':{'VOC_coefficient_range':[float(lopo.VOC_coefficient.min()),float(lopo.VOC_coefficient.max())],'VOC_CI_excludes_zero_positive_count':int(lopo.VOC_ci_low.gt(0).sum()),'n_omissions':len(lopo),'joint_approx_p_range':[float(lopo.joint_F_approx_p.min()),float(lopo.joint_F_approx_p.max())]},'first_differences':first,'first_difference_note':'Consecutive retained reports7–14 days apart, shared calendar-week controls, change in earlier stress controls. Still subject to confounding, short-panel dynamic bias, exposure error and reverse causation. Not a causal effect estimate.'}
    results['first_difference_note']='Consecutive retained reports7–14 days apart. Calendar-week dummy vectors and all three prior-stress controls are differenced exactly. Unmeasured confounding, short-panel dynamic bias, exposure error and reverse causation remain. Not a causal effect estimate.'
    (OUT/'association_sensitivity.json').write_text(json.dumps(results,indent=2))
    report=json.loads((OUT/'results.json').read_text());report['sensitivity']=results;report['sensitivity_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();report['renderer_code_sha256']=hashlib.sha256(Path(main.__file__).read_bytes()).hexdigest();(OUT/'results.json').write_text(json.dumps(report,indent=2));main.render(report)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':run()
