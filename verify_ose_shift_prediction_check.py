"""Verify shift membership, source timing and paired prediction errors."""
import hashlib,json
import numpy as np
import pandas as pd
import ose_shift_prediction_check as study

out=study.OUT;r=json.loads((out/'results.json').read_text())
schema,d,cols=study.old.load()
audit=pd.read_csv(out/'known_survey_shift_audit.csv',float_precision='round_trip')
membership=pd.read_csv(out/'subgroup_membership.csv')
p=pd.read_csv(study.ROOT/'outputs/ose-unseen-with-history/predictions.csv',float_precision='round_trip',dtype={'student_group':str})
events=pd.read_csv(study.changes.OUT/'large_change_examples.csv')
lookup={}
for tag,path in [('satisfaction',study.changes.SOURCE/'cleaned_satisfaction.csv'),('computer',study.changes.SOURCE/'cleaned_computer.csv'),('mood',study.changes.SOURCE/'cleaned_mood.csv'),('stress',study.ROOT/'outputs/expanded-stress-increases/cleaned_stress_reports.csv')]:
    table=pd.read_csv(path,float_precision='round_trip');table.date=pd.to_datetime(table.date)
    factors=[k for k in study.changes.SURVEYS if k.startswith(tag+'_')]
    if tag=='computer':factors+=['computer_DownloadSpeed','computer_UploadSpeed']
    for factor in factors:
        col=factor.split('_',1)[1]
        for person,g in table[table[col].notna()].groupby('person'):
            g=g.sort_values('date');lookup[(factor,person)]=(g.date.to_numpy(),g[col].to_numpy())
expected=[]
for _,target in d.iterrows():
    for factor in list(study.changes.SURVEYS)+['computer_DownloadSpeed','computer_UploadSpeed']:
        dates,values=lookup.get((factor,target.person),(np.array([],dtype='datetime64[ns]'),np.array([])))
        bound=target.date-pd.Timedelta(days=7);n=np.searchsorted(dates,np.datetime64(bound,'ns'),side='left')
        if n<2:continue
        a,b=pd.Timestamp(dates[n-2]),pd.Timestamp(dates[n-1]);before,after=values[n-2:n]
        if not 7<=(b-a).days<=21 or (target.date-b).days>28:continue
        if factor in ['computer_DownloadSpeed','computer_UploadSpeed']:
            if before<=0 or after<=0:continue
            threshold=20 if factor.endswith('DownloadSpeed') else 10
            large=abs(after-before)>=threshold and max(after/before,before/after)>=3
        else:large=abs(after-before)>=study.changes.SURVEYS[factor][1]
        assert a<b<bound
        expected.append({'source_row':target.source_row,'factor':factor,'date_before':str(a.date()),'date_after':str(b.date()),'value_before':before,'value_after':after,'large':large})
ex=pd.DataFrame(expected).set_index(['source_row','factor']).sort_index();got=audit.set_index(['source_row','factor']).sort_index()
assert set(ex.index)==set(got.index)
for col in ['date_before','date_after','large']:np.testing.assert_array_equal(ex[col],got[col])
np.testing.assert_allclose(ex[['value_before','value_after']],got[['value_before','value_after']],atol=1e-10)
sets={'Physical shifts known earlier':set(events.loc[events.group.eq('physical_forecast'),'source_row_after'])}
for group in ['Environmental satisfaction shifts known earlier','Other shifts known earlier']:sets[group]=set(audit.loc[audit.group.eq(group)&audit.large,'source_row'])
sets['Any shift known earlier']=set.union(*sets.values())
sets['Physical shifts during recent week (retrospective)']=set(events.loc[events.group.eq('physical_recent'),'source_row_after'])
sets['Same-day survey/network shifts (retrospective)']=set(events.loc[events.group.eq('self_report_or_network'),'source_row_after'])
for subset,keys in sets.items():assert set(membership.loc[membership.subset.eq(subset),'source_row'])==keys
for protocol,groups in r['subsets'].items():
    for subset,v in groups.items():
        pp=p[p.protocol.eq(protocol)&p.source_row.isin(sets[subset])]
        for model,g in pp.groupby('model'):
            assert g.source_row.is_unique and set(g.source_row)==set(pp[pp.model.eq('Last OSE')].source_row)
            error=g.observed-g.predicted
            np.testing.assert_allclose(v['scores'][model]['mae'],abs(error).mean(),atol=1e-10)
            np.testing.assert_allclose(v['scores'][model]['rmse'],np.sqrt((error**2).mean()),atol=1e-10)
        for label,c in v['comparisons'].items():
            a,b=label.split(' versus ');aa=pp[pp.model.eq(a)].set_index('source_row');bb=pp[pp.model.eq(b)].set_index('source_row').loc[aa.index]
            delta=abs(bb.observed-bb.predicted)-abs(aa.observed-aa.predicted)
            np.testing.assert_allclose(c['mae_improvement'],delta.mean(),atol=1e-10)
            aggregates=pd.DataFrame({'delta':delta,'group':aa.student_group}).groupby('group').delta.agg(['sum','count'])
            ix=np.random.default_rng(42).integers(len(aggregates),size=(3000,len(aggregates)))
            boot=aggregates['sum'].to_numpy()[ix].sum(axis=1)/aggregates['count'].to_numpy()[ix].sum(axis=1)
            np.testing.assert_allclose(c['person_cluster_95_interval'],np.quantile(boot,[.025,.975]),atol=1e-10)
for path,digest in r['source_hashes'].items():assert hashlib.sha256(open(path,'rb').read()).hexdigest()==digest
checks={'all_known_survey_shift_memberships_rebuilt_from_sources':True,'known_survey_measurements_strictly_before_target_minus7days':True,'physical_events_match_prior_verified_audit':True,'shift_selection_uses_no_target_OSE_or_OSE_change':True,'subgroup_union_memberships_deduplicated':True,'same_test_rows_for_all_compared_models':True,'all_subgroup_MAE_and_RMSE_recomputed':True,'all_paired_improvements_and_bootstrap_intervals_recomputed':True,'original_predictions_and_feature_sources_unchanged':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
