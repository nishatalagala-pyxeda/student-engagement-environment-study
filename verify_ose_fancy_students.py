"""Verify nested group splits, selection, fitted transforms and predictions."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
import ose_fancy_students as study

out=study.OUT;schema,d,cols=study.old.load();indexed=d.set_index('source_row',drop=False)
r=json.loads((out/'results.json').read_text())
p=pd.read_csv(out/'predictions.csv',float_precision='round_trip',dtype={'student_group':str})
m=pd.read_csv(out/'split_membership.csv',dtype={'student_group':str})
t=pd.read_csv(out/'inner_tuning.csv',float_precision='round_trip')
selections=json.loads((out/'selected_configurations.json').read_text())
prior=pd.read_csv(study.old.OUT/'predictions.csv',float_precision='round_trip');prior=prior[prior.protocol.eq('All students')]
assert not any(c.startswith('ose_') for c in cols)
for name,digest in r['plan']['source_hashes'].items():assert hashlib.sha256((study.old.SOURCE/name).read_bytes()).hexdigest()==digest
for fold,mm in m.groupby('outer_fold'):
    outer=mm[mm.inner_fold.eq(0)];tr=outer[outer.role.eq('train')];te=outer[outer.role.eq('test')]
    assert set(tr.student_group).isdisjoint(te.student_group)
    assert set(tr.person).isdisjoint(te.person)
    assert set(tr.source_row)|set(te.source_row)==set(d.source_row)
    reference=prior[prior.fold.eq(fold)&prior.model.eq('Random forest')]
    assert set(reference.source_row)==set(te.source_row)
    validation=[]
    for inner,im in mm[mm.inner_fold.gt(0)].groupby('inner_fold'):
        a=im[im.role.eq('train')];b=im[im.role.eq('validation')]
        assert set(a.student_group).isdisjoint(b.student_group)
        assert set(a.source_row)|set(b.source_row)==set(tr.source_row)
        assert set(im.source_row).isdisjoint(te.source_row)
        validation.extend(b.source_row)
        logged=t[t.outer_fold.eq(fold)&t.inner_fold.eq(inner)]
        assert set(logged.validation_n)=={len(b)}
    assert len(validation)==len(set(validation)) and set(validation)==set(tr.source_row)
    models=joblib.load(out/f'fold_{fold}_models.joblib');train=indexed.loc[tr.source_row];test=indexed.loc[te.source_row]
    inner_best={}
    for family,configs in study.CONFIGS.items():
        tt=t[t.outer_fold.eq(fold)&t.family.eq(family)].copy();tt['absolute_sum']=tt.mae*tt.validation_n
        ranked=tt.groupby('config_index').absolute_sum.sum()/len(train)
        best_index=int(ranked.idxmin());inner_best[family]=ranked.loc[best_index]
        selected=next(v for v in selections if v['outer_fold']==fold and v['family']==family)
        assert selected['config_index']==best_index and selected['config']==configs[best_index]
        np.testing.assert_allclose(selected['inner_mae'],ranked.loc[best_index],atol=1e-10)
        bundle=models['models'][family];assert bundle['config']==selected['config']
        pre=bundle['preprocessor'];model=bundle['estimator']
        pp=p[p.fold.eq(fold)&p.model.eq(family)].set_index('source_row').loc[test.source_row]
        np.testing.assert_allclose(pp.predicted,model.predict(pre.transform(test[cols])),atol=1e-10)
        if family=='CatBoost':
            np.testing.assert_allclose(pre.facial.imputer_.statistics_,train[schema['face_features']].median().fillna(0),atol=1e-10)
            assert model.get_params()['iterations']==400
        else:
            num=next(c for n,_,c in pre.transformers_ if n=='numeric')
            np.testing.assert_allclose(pre.named_transformers_['numeric'].steps[0][1].statistics_,train[num].median().fillna(0),atol=1e-10)
            np.testing.assert_allclose(pre.named_transformers_['facial'].imputer_.statistics_,train[schema['face_features']].median().fillna(0),atol=1e-10)
            if family=='Histogram boosting':assert model.early_stopping is False
    assert models['selected_family']==min(inner_best,key=inner_best.get)
    selector=p[p.fold.eq(fold)&p.model.eq('Nested family selection')].set_index('source_row')
    family=p[p.fold.eq(fold)&p.model.eq(models['selected_family'])].set_index('source_row').loc[selector.index]
    np.testing.assert_array_equal(selector.predicted,family.predicted)
for name,g in p.groupby('model'):
    assert g.source_row.is_unique and set(g.source_row)==set(d.source_row)
    np.testing.assert_array_equal(g.observed,indexed.loc[g.source_row].OSE)
    error=g.observed-g.predicted;v=r['scores'][name]
    np.testing.assert_allclose(v['mae'],abs(error).mean(),atol=1e-10)
    np.testing.assert_allclose(v['rmse'],np.sqrt((error**2).mean()),atol=1e-10)
    if name in set(prior.model):
        original=prior[prior.model.eq(name)].set_index('source_row').loc[g.source_row]
        np.testing.assert_array_equal(g.predicted,original.predicted)
for label,v in r['comparisons'].items():
    a,b=label.split(' versus ');aa=p[p.model.eq(a)].set_index('source_row');bb=p[p.model.eq(b)].set_index('source_row').loc[aa.index]
    np.testing.assert_allclose(v['mae_improvement'],(abs(bb.observed-bb.predicted)-abs(aa.observed-aa.predicted)).mean(),atol=1e-10)
checks={'exact_same_outer_test_students_and_records_as_previous_experiment':True,'no_student_group_overlap_in_outer_or_inner_splits':True,'outer_test_rows_never_in_inner_training_or_validation':True,'each_outer_training_record_validated_once_for_tuning':True,'configurations_reselected_independently_from_inner_scores':True,'nested_family_choice_uses_inner_scores_only':True,'all_outer_predictions_reproduce_from_saved_models':True,'imputation_statistics_match_training_students_only':True,'histogram_boosting_random_early_stopping_disabled':True,'no_past_OSE_predictors':True,'source_feature_hashes_unchanged':True,'all_models_cover_same1530_reports_and_source_targets':True,'previous_predictions_reused_exactly':True,'metrics_and_paired_improvements_recomputed':True}
(out/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
