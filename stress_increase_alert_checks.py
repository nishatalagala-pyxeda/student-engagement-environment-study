"""Post-result descriptive alert tradeoffs; no threshold selection or new fits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
import stress_increases as main
OUT=Path(__file__).resolve().parent/'outputs/stress-increases'
p=pd.read_csv(OUT/'predictions.csv');r=json.loads((OUT/'results.json').read_text())
models=['History | linear','History | forest','History + controls + environment | linear'];tradeoffs=[];ranking={}
for protocol,pp in p[p.task.eq('classification')].groupby('protocol'):
    ranking[protocol]={}
    for name in models:
        a=pp[pp.model.eq(name)];labels=sorted(a.person.unique());groups=[a[a.person.eq(x)] for x in labels];rng=np.random.default_rng(42);aucs=[];aps=[]
        for _ in range(3000):
            b=pd.concat([groups[j] for j in rng.integers(len(groups),size=len(groups))],ignore_index=True)
            if b.observed.nunique()==2:aucs.append(roc_auc_score(b.observed,b.predicted));aps.append(average_precision_score(b.observed,b.predicted))
        ranking[protocol][name]={'auc_person_cluster_95_interval':np.quantile(aucs,[.025,.975]).tolist(),'AP_person_cluster_95_interval':np.quantile(aps,[.025,.975]).tolist(),'bootstrap_draws_with_both_classes':len(aucs),'note':'Post-result fixed-prediction bootstrap, not full refit/selection uncertainty.'}
        for threshold in [.1,.2,.3,.5]:
            detected=a.predicted.ge(threshold);y=a.observed.eq(1);tp=int((detected&y).sum());fp=int((detected&~y).sum());fn=int((~detected&y).sum())
            tradeoffs.append({'protocol':protocol,'model':name,'probability_threshold':threshold,'n':len(a),'events':int(y.sum()),'alerts':int(detected.sum()),'true_alerts':tp,'false_alerts':fp,'missed_events':fn,'recall':tp/int(y.sum()),'precision':tp/int(detected.sum()) if detected.sum() else 0.})
pd.DataFrame(tradeoffs).to_csv(OUT/'descriptive_alert_thresholds.csv',index=False)
r['post_result_alert_diagnostics']={'note':'Descriptive thresholds and ranking intervals added after inspecting primary results. No threshold optimized or chosen for deployment. Main definitions/results unchanged.','ranking_intervals':ranking,'tradeoffs':tradeoffs}
(OUT/'results.json').write_text(json.dumps(r,indent=2));main.render(r)
print(json.dumps(ranking,indent=2));print(pd.DataFrame(tradeoffs).query('protocol=="purged_rolling_blocks"').to_string(index=False))
