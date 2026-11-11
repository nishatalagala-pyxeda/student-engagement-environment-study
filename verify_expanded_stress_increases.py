"""Independent source-based checks of expanded and matched forecasting results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import expanded_stress_increases as analysis
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/expanded-stress-increases'
raw=pd.read_csv(next(analysis.old.base.SOURCE.rglob('PANAS_PSS.csv')));raw['source_row']=raw.index+2;raw['person']=raw.ID.map(analysis.old.canonical);raw['date']=pd.to_datetime(raw.StartDate,format='mixed',errors='coerce')
h=raw[raw.person.notna()&raw.date.notna()&raw.PSS.between(0,4)].copy();counts=h.groupby(['person','date']).PSS.nunique();valid=set(counts[counts.eq(1)].index)
h=h[[tuple(x) in valid for x in h[['person','date']].to_numpy()]].drop_duplicates(['person','date']);byid={k:g.sort_values('date') for k,g in h.groupby('person')};byrow=raw.set_index('source_row')
r=json.loads((OUT/'results.json').read_text());audit_count=0;checked_keys=set()
for section,prefix in [('expanded','expanded'),('controlled','controlled')]:
    a=pd.read_csv(OUT/f'{prefix}_history_target_audit.csv');p=pd.read_csv(OUT/f'{prefix}_predictions.csv');audit_count+=len(a)
    for _,row in a.iterrows():
        key=(row.person,row.date,row.role=='test',row.cutoff_exclusive if row.role=='test' else '')
        if key in checked_keys:continue
        checked_keys.add(key);date=pd.Timestamp(row.date);older=byid[row.person];older=older[older.date<date-pd.Timedelta(days=7)]
        if row.role=='test':older=older[older.date<pd.Timestamp(row.cutoff_exclusive)]
        assert len(older)>=2 and len(older)==row.prior_report_count
        assert set(map(int,row.history_source_rows.split(',')))==set(older.source_row) and row.source_row not in set(older.source_row)
        mu=older.PSS.sum()/len(older);actual=byrow.loc[row.source_row,'PSS'];assert byrow.loc[row.source_row,'person']==row.person
        np.testing.assert_allclose(row.past_mean_stress,mu,atol=1e-12,rtol=0);np.testing.assert_allclose(row.excess_stress,actual-mu,atol=1e-12,rtol=0);assert bool(row.event)==bool(actual-mu>=.5)
        np.testing.assert_allclose(row.prior_stress,older.iloc[-1].PSS,atol=1e-12);np.testing.assert_allclose(row.past_sd,np.sqrt(np.mean((older.PSS-mu)**2)),atol=1e-12)
    for (protocol,pool,task),pp in p.groupby(['protocol','training_pool','task']):
        sets=[set(g.source_row) for _,g in pp.groupby('model')];assert all(s==sets[0] for s in sets)
        audit=a[a.protocol.eq(protocol)&a.training_pool.eq(pool)&a.role.eq('test')].set_index('source_row');assert audit.index.is_unique
        for name,g in pp.groupby('model'):
            assert g.source_row.is_unique;ref=audit.loc[g.source_row];metric=r[section]['scores'][protocol][pool][task][name]
            if task=='classification':
                np.testing.assert_array_equal(g.observed.to_numpy(),ref.event.astype(float).to_numpy());np.testing.assert_allclose(metric['brier'],np.mean((g.predicted-g.observed)**2),atol=1e-12);np.testing.assert_allclose(metric['roc_auc'],roc_auc_score(g.observed,g.predicted),atol=1e-12)
            else:
                np.testing.assert_allclose(g.observed.to_numpy(),ref.excess_stress.to_numpy(),atol=1e-12);np.testing.assert_allclose(metric['mae'],np.mean(np.abs(g.predicted-g.observed)),atol=1e-12);assert (g.predicted+g.history_mean).between(-1e-12,4+1e-12).all()
cp=pd.read_csv(OUT/'controlled_predictions.csv')
for (protocol,task,name),g in cp.groupby(['protocol','task','model']):
    orig=g[g.training_pool.eq('Original training')].set_index('source_row').sort_index();expanded=g[g.training_pool.eq('Expanded training')].set_index('source_row').sort_index();assert orig.index.equals(expanded.index)
    np.testing.assert_allclose(orig.observed,expanded.observed,atol=1e-12);np.testing.assert_allclose(orig.history_mean,expanded.history_mean,atol=1e-12)
    saved=pd.read_csv(ROOT/'outputs/stress-increases/predictions.csv');saved=saved[saved.protocol.eq(protocol)&saved.task.eq(task)&saved.model.eq(name)].set_index('source_row').sort_index();assert orig.index.equals(saved.index);np.testing.assert_allclose(orig.predicted,saved.predicted,rtol=1e-10,atol=1e-10)
# Paired AUC differences for the same original test rows, as specified in the plan.
auc_intervals={}
for (protocol,name),g in cp[cp.task.eq('classification')&cp.model.str.startswith('History')].groupby(['protocol','model']):
    orig=g[g.training_pool.eq('Original training')].set_index('source_row').sort_index();expanded=g[g.training_pool.eq('Expanded training')].set_index('source_row').loc[orig.index]
    labels=orig.person.to_numpy();groups=[np.flatnonzero(labels==x) for x in np.unique(labels)];rng=np.random.default_rng(42);yy=orig.observed.to_numpy();aa=expanded.predicted.to_numpy();bb=orig.predicted.to_numpy();boot=[]
    for _ in range(3000):
        ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(yy[ix]))==2:boot.append(roc_auc_score(yy[ix],aa[ix])-roc_auc_score(yy[ix],bb[ix]))
    auc_intervals.setdefault(protocol,{})[name]={'auc_improvement':float(roc_auc_score(yy,aa)-roc_auc_score(yy,bb)),'person_cluster_95_interval':np.quantile(boot,[.025,.975]).tolist(),'draws':len(boot)}
r['controlled_auc_improvement_intervals']=auc_intervals
(OUT/'results.json').write_text(json.dumps(r,indent=2));analysis.render(r)
checks={'cleaned_history_reconstructed_independently_from_raw_source':True,'all_distinct_historical_reference_targets_and_features_recomputed':True,'all_primary_auc_brier_mae_recomputed':True,'all_prediction_bounds_valid':True,'all_models_use_identical_test_rows_within_comparison':True,'controlled_training_test_rows_and_reference_means_identical':True,'original_training_predictions_reproduce_previous_saved_results':True,'audited_rows':audit_count,'distinct_history_reference_keys':len(checked_keys)}
(OUT/'independent_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2));print(json.dumps(auc_intervals,indent=2))
