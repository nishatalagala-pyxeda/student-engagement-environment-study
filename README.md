# Student engagement, emotion, and ordinary environmental variation

Exploratory research using InGauge/EnGage and DEPRESSData to compare personal history with environmental, behavioral, and psychological predictors of engagement and emotion.

## Hypothesis

Students' engagement exhibits temporal persistence: previous engagement predicts subsequent engagement more accurately than ordinary fluctuations in environmental conditions, stress, or behavior. Additional measurements are expected to provide limited incremental predictive value once engagement history is known.

These are preliminary, directional experiments. A student should independently reproduce the preparation, verify measurements and timing, and specify evaluation criteria before repeating or extending the comparisons. Reusing these datasets does not constitute independent external validation.

## Why the project matters

Sensors, wearable devices, and repeated surveys impose costs and collect sensitive information. Establishing whether they improve on simple historical baselines can help determine which measurements are useful. Evaluation must distinguish persistent differences between students from prediction of meaningful within-student changes.

## Datasets

- **InGauge/EnGage:** classroom conditions and air-conditioning operation in a secondary school near Melbourne, Australia, together with wearable physiology and surveys of emotion, comfort, and engagement. The EnGage study involved 23 students and six teachers. Incomplete classroom-to-sensor mapping limits the local environmental analyses.
- **DEPRESSData:** COVID-era online learning, with repeated engagement and psychological surveys, bedroom environmental sensors, environmental satisfaction, computer/network performance, diaries, Fitbit data, facial feature summaries, and demographics. The cleaned OSE data contain 1,530 reports from 158 cohort-labelled student IDs; modality coverage differs considerably.

**OSE interpretation:** the supplied questionnaire matches Online Student Engagement, despite the dataset README's self-efficacy label. Item-level scoring is unavailable; models use the supplied aggregate without inventing theoretical limits or clinical interpretations.

Acquire source data separately under the dataset providers' terms and cite the original publications. InGauge/EnGage: [dataset](https://rmit.figshare.com/articles/dataset/In-Gauge_and_En-Gage_Datasets/14578908), [dataset paper](https://doi.org/10.1038/s41597-022-01347-w), and [author repository](https://github.com/cruiseresearchgroup/InGauge-and-EnGage-Datasets). The supplied DEPRESSData documentation names its accompanying paper *Dataset on Emotions, Performance, Responses, Environment, and Satisfaction of Students in COVID-19 E-learning*; consult that release for its complete citation.

## Directional results

- On the same 1,118 history-eligible DEPRESSData reports from 138 student IDs, entirely held-out-student evaluation gave MAE **4.10** for ridge using OSE history alone. An all-variable forest improved from **8.29 without past OSE to 4.57 with it**, but did not outperform simpler history-based predictions.
- On the broader 1,530-report held-out-student cohort, without past OSE, tuned Elastic Net achieved MAE **7.98**, compared with **9.01** for the training-population mean. This reflects all non-OSE predictors together, not environment alone, and is a different population from the history-eligible comparison.
- Updating non-OSE inputs to current measurements and adding explicit changes did not establish a reliable incremental benefit. Across 697 contemporaneous substantial-shift cases, matched Elastic Net MAE was **4.22 for history alone versus 4.33 for all current predictors and changes**. A smaller physical-shift subgroup showed an uncertain improvement, **3.84 to 3.64**, on 61 reports from 26 IDs.
- Across 1,085 within-student paired observations, stress change and OSE change had correlation **−0.093**. Current stress did not improve OSE prediction once history was included.
- InGauge/EnGage emotion and engagement models showed limited reliable improvement over simple baselines under the available sample and alignment constraints.

These findings concern predictive value under observed conditions, not absence of environmental or psychological influence. Stress is a psychological measure, rather than a physical environmental exposure. Several shift thresholds are illustrative, and ordinary changes should not be described as major disruptions.

## Evaluation principles

- Keep entire students out of training when evaluating transfer to unseen students. Identical numeric ID suffixes across DEPRESSData cohorts are conservatively grouped because physical identity is unverified.
- A held-out student can supply earlier personal OSE at inference without contributing training outcomes. This differs from predicting for someone with no OSE history.
- Fit imputation, encoding, scaling, and facial PCA only inside each training fold; tune inside nested student-held-out folds.
- Distinguish chronological forecasting from concurrent estimation. Current questionnaires are concurrent inputs and may share response bias with the outcome; they cannot be described as week-ahead predictors.
- Compare models on identical records. Do not compare raw errors across different coverage/eligibility cohorts as if the difference were an algorithm effect.
- Treat source timing, inferred sensor mappings, sparse modalities, student dependence, and repeated exploratory test reuse as material limitations. Bootstrap intervals are conditional on fixed predictions.

## Code and reproduction

[README-model.md](README-model.md) records the experiment sequence, preparation rules, results, and dependencies between scripts. The root Python scripts contain preparation, modeling, and independent verification routines. Generated HTML reports, feature tables, predictions, and saved fold models are written under `outputs/` locally.

The experiments currently assume the original local source layout in Downloads and, for some scripts, the original macOS paths and temporary dependency directories. **This is an exploratory code archive, not a one-command portable package.** Review the `SOURCE`, input defaults, and local paths in the preparation scripts before running elsewhere. Some scripts depend on outputs from earlier experiments and cached archive summaries; consult the detailed notes rather than running scripts alphabetically.

Use Python 3.12 and the pinned analysis dependencies in `requirements-model.txt`. CatBoost experiments additionally require `requirements-catboost.txt`. Reading source workbooks and questionnaire images also requires the applicable document/image libraries used by the preparation scripts. Local execution used single-thread BLAS/OMP settings and `JOBLIB_MULTIPROCESSING=0`; large facial archives were streamed rather than fully extracted.

Verification scripts are named `verify_*.py`; they reconstruct source features, student splits, temporal cutoffs, preprocessing statistics, held-out predictions, and reported metrics for their associated experiments. Run them after generating the required inputs and model outputs.

## Repository contents and data handling

This repository includes analysis code, verification scripts, dependency specifications, and aggregate documentation. Raw datasets, downloaded reference material, participant-level derived data, reports containing participant examples, and trained models are intentionally excluded by `.gitignore`. Those local files are not deleted. Do not commit them without separately checking the providers' redistribution terms and the information they contain.
