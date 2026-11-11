"""Check held-out isolation, personal-history chronology and saved forecasts."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
import ose_unseen_with_history as study

out=study.OUT;schema,d,other=study.old.load();indexed=d.set_index('source_row',drop=False)
r=json.loads((out/'results.json').read_text())
p=pd.read_csv(out/'predictions.csv',float_precision='round_trip',dtype={'student_group':str})
m=pd.read_csv(out/'fold_membership.csv',dtype={'student_group':str})
original=pd.read_csv(study.old.OUT/'predictions.csv',float_precision='round_trip')
raw=pd.read_csv(study.old.ROOT/'outputs/ose-stress-prediction/cleaned_ose.csv',float_precision='round_trip');raw.date=pd.to_datetime(raw.date)
histcols=study.old.previous.core.OSE_FEATURES;reconstructed=[];audit=[]
for _,row in d.iterrows():
    h=raw[raw.person.eq(row.person)&raw.date.lt(row.date-pd.Timedelta(days=7))].sort_values('date');v=h.OSE.to_numpy();n=len(v)
    if n:
        mean=np.mean(v);values=[mean,v[-1],v[-1]-mean,np.std(v),v[-1]-v[-2] if n>1 else np.nan,n,(row.date-h.iloc[-1].date).days]
        assert row.source_row not in set(h.source_row)
    else:values=[np.nan,np.nan,np.nan,np.nan,np.nan,0,np.nan]
    reconstructed.append(values)
    audit.append({'source_row':int(row.source_row),'person':row.person,'date':str(row.date.date()),'bound_exclusive':str((row.date-pd.Timedelta(days=7)).date()),'history_n':n,'latest_date':str(h.iloc[-1].date.date()) if n else None,'history_source_rows':','.join(map(str,h.source_row))})
np.testing.assert_allclose(d[histcols].to_numpy(float),np.array(reconstructed),equal_nan=True,atol=1e-10)
pd.DataFrame(audit).to_csv(out/'personal_history_audit.csv',index=False)
for name,digest in r['plan']['source_hashes'].items():assert hashlib.sha256((study.old.SOURCE/name).read_bytes()).hexdigest()==digest
for (protocol,fold),mm in m.groupby(['protocol','fold']):
    tr=mm[mm.role.eq('train')];te=mm[mm.role.eq('test')]
    assert set(tr.person).isdisjoint(te.person) and set(tr.student_group).isdisjoint(te.student_group)
    train=indexed.loc[tr.source_row];test=indexed.loc[te.source_row]
    reference=original[original.protocol.eq(protocol)&original.fold.eq(fold)&original.model.eq('Random forest')]
    assert set(reference.source_row)==set(test.source_row)
    path=out/(('all' if protocol=='All students' else 'eligible')+f'_fold_{fold}.joblib');models=joblib.load(path)
    for name,bundle in models.items():
        pp=p[p.protocol.eq(protocol)&p.fold.eq(fold)&p.model.eq(name)].set_index('source_row').loc[test.source_row]
        cols=bundle['features'];assert 'OSE' not in cols and not set(cols)&{'source_row','person','student_group','date','eligible'}
        model=bundle['model'];np.testing.assert_allclose(pp.predicted,model.predict(test[cols]),atol=1e-10)
        pre=model.steps[0][1];nums=next(c for n,_,c in pre.transformers_ if n=='numeric')
        np.testing.assert_allclose(pre.named_transformers_['numeric'].steps[0][1].statistics_,train[nums].median().fillna(0),atol=1e-10)
    for name,col in [('Last OSE','ose_last'),('Past mean OSE','ose_mean')]:
        pp=p[p.protocol.eq(protocol)&p.fold.eq(fold)&p.model.eq(name)].set_index('source_row').loc[test.source_row]
        np.testing.assert_allclose(pp.predicted,test[col].fillna(train.OSE.mean()),atol=1e-10)
for protocol,pp in p.groupby('protocol'):
    expected=d[d.eligible] if protocol=='Previous eligible subset' else d
    for name,g in pp.groupby('model'):
        assert g.source_row.is_unique and set(g.source_row)==set(expected.source_row)
        np.testing.assert_array_equal(g.observed,indexed.loc[g.source_row].OSE)
        np.testing.assert_allclose(r['scores'][protocol][name]['mae'],abs(g.predicted-g.observed).mean(),atol=1e-10)
        np.testing.assert_allclose(r['scores'][protocol][name]['rmse'],np.sqrt(((g.predicted-g.observed)**2).mean()),atol=1e-10)
        oldname=name.replace('Without past OSE | ','')
        ref=original[original.protocol.eq(protocol)&original.model.eq(oldname)]
        if len(ref):np.testing.assert_array_equal(g.predicted,ref.set_index('source_row').loc[g.source_row].predicted)
    for label,v in r['comparisons'][protocol].items():
        a,b=label.split(' versus ');aa=pp[pp.model.eq(a)].set_index('source_row');bb=pp[pp.model.eq(b)].set_index('source_row').loc[aa.index]
        np.testing.assert_allclose(v['mae_improvement'],(abs(bb.observed-bb.predicted)-abs(aa.observed-aa.predicted)).mean(),atol=1e-10)
checks={'all1530_personal_OSE_histories_rebuilt_from_cleaned_source':True,'current_and_future_OSE_never_history_inputs':True,'strict_target_minus7day_history_cutoff':True,'no_held_out_students_or_identity_groups_in_training':True,'exact_prior_student_splits_and_without_OSE_predictions':True,'each_model_tests_identical_records_once_per_protocol':True,'saved_models_reproduce_predictions':True,'imputation_statistics_match_training_students_only':True,'last_and_mean_OSE_baselines_and_fallbacks_recomputed':True,'all_MAE_RMSE_and_paired_improvements_recomputed':True,'source_hashes_unchanged':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
