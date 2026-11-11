"""Verify nested student isolation, preprocessing, predictions and subgroup metrics."""
import json
import joblib
import numpy as np
import pandas as pd
import ose_current_shift_models as study

out=study.OUT;d,schema,scopes,metadata=study.load();indexed=d.set_index('source_row',drop=False)
r=json.loads((out/'results.json').read_text());p=pd.read_csv(out/'predictions.csv',float_precision='round_trip',dtype={'student_group':str})
outer=pd.read_csv(out/'outer_membership.csv',dtype={'student_group':str});inner=pd.read_csv(out/'inner_membership.csv',dtype={'student_group':str});tuning=pd.read_csv(out/'inner_tuning.csv',float_precision='round_trip');selected=json.loads((out/'selected_alphas.json').read_text())
old_members=pd.read_csv(study.old.OUT/'fold_membership.csv',dtype={'student_group':str});old_members=old_members[old_members.protocol.eq('Previous eligible subset')]
pd.testing.assert_frame_equal(outer.reset_index(drop=True),old_members.reset_index(drop=True))
previous=pd.read_csv(study.ROOT/'outputs/ose-unseen-with-history/predictions.csv',float_precision='round_trip');previous=previous[previous.protocol.eq('Previous eligible subset')]
held=outer[outer.role.eq('test')];assert held.source_row.is_unique and set(held.source_row)==set(d.loc[d.eligible,'source_row'])
for fold,mm in outer.groupby('fold'):
    tr=mm[mm.role.eq('train')];te=mm[mm.role.eq('test')]
    assert set(tr.person).isdisjoint(te.person) and set(tr.student_group).isdisjoint(te.student_group)
    train=indexed.loc[tr.source_row];test=indexed.loc[te.source_row];validation=[]
    for inner_fold,im in inner[inner.outer_fold.eq(fold)].groupby('inner_fold'):
        a=im[im.role.eq('train')];b=im[im.role.eq('validation')]
        assert set(a.student_group).isdisjoint(b.student_group)
        assert set(a.source_row)|set(b.source_row)==set(tr.source_row)
        assert set(im.source_row).isdisjoint(te.source_row);validation.extend(b.source_row)
        tt=tuning[tuning.outer_fold.eq(fold)&tuning.inner_fold.eq(inner_fold)];assert set(tt.validation_n)=={len(b)}
    assert len(validation)==len(set(validation)) and set(validation)==set(tr.source_row)
    models=joblib.load(out/f'fold_{fold}_models.joblib')
    for scope,cols in scopes.items():
        tt=tuning[tuning.outer_fold.eq(fold)&tuning.scope.eq(scope)].copy();tt['absolute_sum']=tt.validation_n*tt.mae
        ranking=tt.groupby('alpha').absolute_sum.sum()/len(train);best=float(ranking.idxmin())
        choice=next(x for x in selected if x['outer_fold']==fold and x['scope']==scope);assert choice['alpha']==best;np.testing.assert_allclose(choice['inner_mae'],ranking.loc[best],atol=1e-10)
        for family in ['Random forest','Elastic Net']:
            name=scope+' | '+family;bundle=models[name];assert bundle['features']==cols
            pp=p[p.fold.eq(fold)&p.model.eq(name)].set_index('source_row').loc[test.source_row]
            pre=bundle['preprocessor'];est=bundle['estimator'];np.testing.assert_allclose(pp.predicted,est.predict(pre.transform(test[cols])),atol=1e-10)
            if family=='Elastic Net':assert est.alpha==best and est.l1_ratio==.9
            else:assert est.n_estimators==300 and est.max_depth==3 and est.min_samples_leaf==10
            nums=next(c for n,_,c in pre.transformers_ if n=='numeric')
            np.testing.assert_allclose(pre.named_transformers_['numeric'].steps[0][1].statistics_,train[nums].median().fillna(0),atol=1e-10)
            if 'facial' in pre.named_transformers_:
                face=next(c for n,_,c in pre.transformers_ if n=='facial')
                np.testing.assert_allclose(pre.named_transformers_['facial'].imputer_.statistics_,train[face].median().fillna(0),atol=1e-10)
            if 'categorical' in pre.named_transformers_:
                cats=next(c for n,_,c in pre.transformers_ if n=='categorical');block=pre.named_transformers_['categorical'];retained=block.steps[0][1].get_feature_names_out(cats);categories=block.steps[-1][1].categories_
                assert len(retained)==len(categories)
                for col,values in zip(retained,categories):assert set(values)==set(train[col].fillna('Unknown'))
for name,g in p.groupby('model'):
    assert g.source_row.is_unique and set(g.source_row)==set(d.loc[d.eligible,'source_row'])
    np.testing.assert_array_equal(g.observed,indexed.loc[g.source_row].OSE)
    if name=='History only | Random forest':
        ref=previous[previous.model.eq('Past OSE only | Random forest')].set_index('source_row').loc[g.source_row];np.testing.assert_allclose(g.predicted,ref.predicted,atol=1e-10)
members=pd.read_csv(out/'subgroup_membership.csv');definitions=study.subset_definitions(d)
for subset,keys in definitions.items():
    assert set(members.loc[members.subset.eq(subset),'source_row'])==keys
    pp=p[p.source_row.isin(keys)];v=r['subsets'][subset]
    for name,g in pp.groupby('model'):
        error=g.observed-g.predicted
        np.testing.assert_allclose(v['scores'][name]['mae'],abs(error).mean(),atol=1e-10);np.testing.assert_allclose(v['scores'][name]['rmse'],np.sqrt((error**2).mean()),atol=1e-10)
    for label,c in v['comparisons'].items():
        a,b=label.split(' versus ');aa=pp[pp.model.eq(a)].set_index('source_row');bb=pp[pp.model.eq(b)].set_index('source_row').loc[aa.index]
        delta=abs(bb.observed-bb.predicted)-abs(aa.observed-aa.predicted);np.testing.assert_allclose(c['mae_improvement'],delta.mean(),atol=1e-10)
        group=pd.DataFrame({'delta':delta,'group':aa.student_group}).groupby('group').delta.agg(['sum','count']);ix=np.random.default_rng(42).integers(len(group),size=(3000,len(group)))
        boot=group['sum'].to_numpy()[ix].sum(axis=1)/group['count'].to_numpy()[ix].sum(axis=1);np.testing.assert_allclose(c['person_cluster_95_interval'],np.quantile(boot,[.025,.975]),atol=1e-10)
checks={'exact_previous_outer_student_folds':True,'outer_and_inner_student_groups_disjoint':True,'outer_targets_never_used_for_tuning':True,'inner_alpha_selection_recomputed':True,'all_saved_models_reproduce_held_out_predictions':True,'train_only_numeric_facial_imputation_and_category_sets_verified':True,'history_forest_matches_previous_experiment':True,'each_model_tests_same1118_records_once':True,'subgroups_match_previous_factor_only_audit':True,'all_MAE_RMSE_and_bootstrap_comparisons_recomputed':True}
(out/'model_independent_checks.json').write_text(json.dumps(checks,indent=2));combined={'features':json.loads((out/'feature_independent_checks.json').read_text()),'models':checks};(out/'independent_checks.json').write_text(json.dumps(combined,indent=2));print(json.dumps(checks,indent=2))
