"""Independently verify held-out membership, preprocessing and saved predictions."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
import ose_unseen_students as study

schema,d,cols=study.load();out=study.OUT
r=json.loads((out/'results.json').read_text())
p=pd.read_csv(out/'predictions.csv',float_precision='round_trip',dtype={'student_group':str})
m=pd.read_csv(out/'fold_membership.csv',dtype={'student_group':str})
payload=joblib.load(out/'fold_models.joblib')
assert cols==payload['features'] and not any(c.startswith('ose_') for c in cols)
for name,digest in r['plan']['source_hashes'].items():assert hashlib.sha256((study.SOURCE/name).read_bytes()).hexdigest()==digest
indexed=d.set_index('source_row',drop=False)
for protocol,expected in [('All students',d),('Previous eligible subset',d[d.eligible])]:
    mm=m[m.protocol.eq(protocol)];held=mm[mm.role.eq('test')]
    assert held.source_row.is_unique and set(held.source_row)==set(expected.source_row)
    assert held.groupby('student_group').fold.nunique().max()==1
    for fold,g in mm.groupby('fold'):
        tr=g[g.role.eq('train')];te=g[g.role.eq('test')]
        assert set(tr.person).isdisjoint(te.person)
        assert set(tr.student_group).isdisjoint(te.student_group)
        assert set(tr.source_row)|set(te.source_row)==set(expected.source_row)
        assert set(tr.source_row).isdisjoint(te.source_row)
        train=indexed.loc[tr.source_row];test=indexed.loc[te.source_row]
        for name in ['Ridge','Random forest','Population mean','Population median']:
            pp=p[p.protocol.eq(protocol)&p.fold.eq(fold)&p.model.eq(name)].set_index('source_row').loc[test.source_row]
            np.testing.assert_array_equal(pp.observed,test.OSE)
            if name in ['Ridge','Random forest']:
                model=payload['models'][(protocol,int(fold),name)]
                np.testing.assert_allclose(pp.predicted,model.predict(test[cols]),atol=1e-10)
                pre=model.steps[0][1]
                numeric=next(c for n,t,c in pre.transformers_ if n=='numeric')
                np.testing.assert_allclose(pre.named_transformers_['numeric'].steps[0][1].statistics_,train[numeric].median().fillna(0).to_numpy(),atol=1e-10)
                facial=pre.named_transformers_['facial']
                np.testing.assert_allclose(facial.imputer_.statistics_,train[schema['face_features']].median().fillna(0).to_numpy(),atol=1e-10)
                cats=next(c for n,t,c in pre.transformers_ if n=='categorical')
                categorical=pre.named_transformers_['categorical']
                encoder=categorical.steps[-1][1]
                retained=categorical.steps[0][1].get_feature_names_out(cats)
                assert len(retained)==len(encoder.categories_)
                for c,categories in zip(retained,encoder.categories_):assert set(categories)==set(train[c].fillna('Unknown'))
            else:
                value=train.OSE.mean() if name=='Population mean' else train.OSE.median()
                np.testing.assert_allclose(pp.predicted,value,atol=1e-10)
    for name,pp in p[p.protocol.eq(protocol)].groupby('model'):
        assert pp.source_row.is_unique and set(pp.source_row)==set(expected.source_row)
        error=pp.observed-pp.predicted;v=r['scores'][protocol][name]
        np.testing.assert_allclose(v['mae'],np.abs(error).mean(),atol=1e-10)
        np.testing.assert_allclose(v['rmse'],np.sqrt((error**2).mean()),atol=1e-10)
        np.testing.assert_allclose(v['equal_student_mae'],np.abs(error).groupby(pp.person).mean().mean(),atol=1e-10)
    for label,v in r['comparisons'][protocol].items():
        a,b=label.split(' versus ');pp=p[p.protocol.eq(protocol)]
        aa=pp[pp.model.eq(a)].set_index('source_row');bb=pp[pp.model.eq(b)].set_index('source_row').loc[aa.index]
        np.testing.assert_allclose(v['mae_improvement'],(abs(bb.observed-bb.predicted)-abs(aa.observed-aa.predicted)).mean(),atol=1e-10)
checks={'no_student_or_conservative_identity_group_overlap':True,'each_report_tested_exactly_once_per_model':True,'all_models_use_identical_test_records':True,'no_past_OSE_predictors':True,'primary_includes_students_without_past_OSE_or_stress':True,'source_feature_hashes_unchanged':True,'all_saved_models_reproduce_held_out_predictions':True,'numerical_and_facial_imputers_match_training_students_only':True,'category_encoders_match_training_students_only':True,'population_baselines_recomputed_from_training_students':True,'metrics_and_paired_improvements_independently_recomputed':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
