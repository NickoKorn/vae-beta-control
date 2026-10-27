"""Descriptive TCGA input analysis; no VAE, model fitting or feature-selection claims.

Feature-wise total/within/between variances and eta-squared use the same
labelled patients. Both modalities are described on their stored
scales, including observed negative RNA values. Features are processed in row batches; both full omics matrices need
not be resident in RAM. Class moments use scipy/pandas on selected features.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import skew, kurtosis

FILES = {'RNA': 'data_mrna_seq_v2_rsem_formatted.parquet',
         'Methylation': 'data_methylation_per_gene_formatted.parquet'}


def clean_labels(values):
    text = values.astype('string').str.strip()
    return text.mask(text.str.lower().isin(['unknown','missing','nan','none','','na','not reported','[not available]']))


def load_tcga_cohort(data_root):
    """Keep RNA/methylation-paired samples; preserve missing clinical labels."""
    data_root = Path(data_root)
    clinical = pd.read_parquet(data_root/'data_clinical_formatted.parquet')
    clinical.index = clinical.index.astype(str)
    overview, indices = [], []
    for modality, filename in FILES.items():
        file = pq.ParquetFile(data_root/filename)
        index = file.schema_arrow.pandas_metadata['index_columns'][0]
        features = [name for name in file.schema_arrow.names if name != index]
        sample_ids = pd.read_parquet(data_root/filename, columns=[features[0]]).index.astype(str)
        if not sample_ids.is_unique:
            raise ValueError(f'Duplicate sample IDs in {filename}')
        indices.append(sample_ids)
        overview.append(dict(modality=modality, samples=len(sample_ids), features=len(features),
                             scale='stored RNA values' if modality=='RNA' else 'stored methylation values'))
    paired = indices[0][indices[0].isin(indices[1])]
    cohort = clinical.reindex(paired).copy()
    cohort['AGE'] = pd.to_numeric(cohort.AGE, errors='coerce')
    for column in ['CANCER_TYPE_ACRONYM','CANCER_TYPE','CANCER_TYPE_DETAILED','SEX',
                   'SUBTYPE','AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT']:
        cohort[column] = clean_labels(cohort[column])
    return cohort, pd.DataFrame(overview)


def task_groups(clinical):
    """Descriptive cohorts, not train/validation probe eligibility rules."""
    cancer = clinical.CANCER_TYPE_ACRONYM
    groups = dict(cancer_type=cancer, cancer_group=clinical.CANCER_TYPE,
                  histology=clinical.CANCER_TYPE_DETAILED, sex_all=clinical.SEX,
                  brca_subtype=clinical.SUBTYPE.where(cancer.eq('BRCA')),
                  ucec_subtype=clinical.SUBTYPE.where(cancer.eq('UCEC')),
                  stage_all=clinical.AJCC_PATHOLOGIC_TUMOR_STAGE_SHORT)
    for name in ('COAD','LUAD','LUSC','READ'):
        groups['sex_'+name] = clinical.SEX.where(cancer.eq(name))
    return groups


def feature_variance_by_task(path, modality, cohort, groups, batch_size=256, patient_summary_path=None):
    """Patient-weighted ANOVA decomposition, with per-feature missingness.

    total_variance = within_variance + between_variance (ddof=0).
    eta_squared = between_variance / total_variance; undefined for constants.
    This describes association with known labels, not causality or held-out
    predictivity. Missing labels are excluded separately for each task.
    """
    file = pq.ParquetFile(path)
    index_column = file.schema_arrow.pandas_metadata['index_columns'][0]
    features = [name for name in file.schema_arrow.names if name != index_column]
    states = {}
    for task, labels in groups.items():
        classes = sorted(labels.dropna().unique())
        states[task] = (classes, np.zeros((len(classes),len(features))),
                        np.zeros((len(classes),len(features))), np.zeros((len(classes),len(features))))
    quality = dict(modality=modality, samples=0, finite_cells=0, missing_cells=0, negative_cells=0, minimum=np.inf, maximum=-np.inf)
    patient_tables = []
    for batch in file.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        frame.index = frame.index.astype(str)
        frame = frame.loc[frame.index.isin(cohort.index), features]
        if frame.empty: continue
        values = frame.to_numpy(dtype=float)
        finite = np.isfinite(values)
        quality['samples'] += len(values)
        quality['finite_cells'] += int(finite.sum())
        quality['missing_cells'] += int((~finite).sum())
        quality['negative_cells'] += int(((values<0)&finite).sum())
        quality['minimum'] = min(quality['minimum'],float(np.min(values[finite])))
        quality['maximum'] = max(quality['maximum'],float(np.max(values[finite])))
        patient_tables.append(pd.DataFrame(dict(sample_id=frame.index, mean=np.nanmean(values,axis=1),
            std=np.nanstd(values,axis=1), minimum=np.nanmin(values,axis=1),maximum=np.nanmax(values,axis=1),
            negative_fraction=((values<0)&finite).sum(axis=1)/np.maximum(finite.sum(axis=1),1),
            missing_fraction=(~finite).mean(axis=1))))
        values = np.where(finite, values, 0.)
        squares = values*values
        for task, labels in groups.items():
            names = labels.reindex(frame.index)
            classes, counts, sums, sumsquares = states[task]
            for i, name in enumerate(classes):
                keep = names.eq(name).fillna(False).to_numpy(dtype=bool)
                if keep.any():
                    counts[i] += finite[keep].sum(axis=0)
                    sums[i] += values[keep].sum(axis=0)
                    sumsquares[i] += squares[keep].sum(axis=0)
    tables = []
    for task, (classes, counts, sums, sumsquares) in states.items():
        n = counts.sum(axis=0); total_sum = sums.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean = total_sum/n
            total_ss = np.maximum(0, sumsquares.sum(axis=0)-total_sum*total_sum/n)
            class_correction = np.divide(sums*sums,counts,out=np.zeros_like(sums),where=counts>0)
            within_ss = np.maximum(0,sumsquares-class_correction).sum(axis=0)
            between_ss = np.maximum(0,total_ss-within_ss)
            eta = np.divide(between_ss,total_ss,out=np.full_like(total_ss,np.nan),where=total_ss>0)
        tables.append(pd.DataFrame(dict(task=task,modality=modality,feature=features,n=n.astype(int),
            n_classes=(counts>0).sum(axis=0),mean=mean,total_variance=total_ss/n,
            within_variance=within_ss/n,between_variance=between_ss/n,eta_squared=eta)))
    result = pd.concat(tables,ignore_index=True)
    result.attrs['value_quality'] = quality
    if patient_summary_path is not None:
        pd.concat(patient_tables,ignore_index=True).to_csv(patient_summary_path,index=False)
    return result


def class_moments(path, modality, cohort, groups, effects, n_features=4):
    """Exact scipy skew/excess-kurtosis and quartiles for illustrative features.

    Select by eta-squared and total variance on the descriptive cohort only.
    This selection must not be reused as independently validated feature ranking.
    """
    selected = {}
    for task, table in effects.groupby('task'):
        selected[task] = list(dict.fromkeys(
            table.dropna(subset=['eta_squared']).nlargest(n_features,'eta_squared').feature.tolist()
            + table.dropna(subset=['total_variance']).nlargest(n_features,'total_variance').feature.tolist()))
    columns = list(dict.fromkeys(name for names in selected.values() for name in names))
    data = pd.read_parquet(path, columns=columns).reindex(cohort.index)
    rows = []
    for task, names in selected.items():
        labels = groups[task]
        for name in labels.dropna().unique():
            frame = data.loc[labels.eq(name).fillna(False),names]
            for feature in names:
                values = frame[feature].dropna().to_numpy(dtype=float)
                variable = len(values)>1 and np.ptp(values)>0
                rows.append(dict(task=task,modality=modality,feature=feature,group=str(name),
                    n=len(values),missing=int(frame[feature].isna().sum()),
                    mean=np.mean(values) if len(values) else np.nan,
                    variance=np.var(values,ddof=1) if len(values)>1 else np.nan,
                    std=np.std(values,ddof=1) if len(values)>1 else np.nan,
                    median=np.median(values) if len(values) else np.nan,
                    q25=np.quantile(values,.25) if len(values) else np.nan,
                    q75=np.quantile(values,.75) if len(values) else np.nan,
                    skewness=skew(values,bias=False) if variable and len(values)>2 else np.nan,
                    excess_kurtosis=kurtosis(values,fisher=True,bias=False) if variable and len(values)>3 else np.nan))
    return pd.DataFrame(rows), data


def run_tcga_data_analysis(data_root, output_dir, n_features=4):
    """Export raw-input cohort counts, feature variance decompositions and moments."""
    data_root, output_dir = Path(data_root), Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    clinical, overview = load_tcga_cohort(data_root)
    groups = task_groups(clinical)
    counts = pd.DataFrame([dict(task=task,group=str(name),n=int(n))
        for task,labels in groups.items() for name,n in labels.value_counts().items()])
    cohort_rows = []
    for task, labels in groups.items():
        mask = pd.Series(True,index=clinical.index)
        if task in ('brca_subtype','ucec_subtype'):
            mask = clinical.CANCER_TYPE_ACRONYM.eq('BRCA' if task=='brca_subtype' else 'UCEC').fillna(False)
        elif task.startswith('sex_') and task != 'sex_all':
            mask = clinical.CANCER_TYPE_ACRONYM.eq(task.removeprefix('sex_')).fillna(False)
        n = int(mask.sum()); known = int(labels[mask].notna().sum())
        cohort_rows.append(dict(task=task,n_cohort=n,n_known=known,n_missing=n-known,
                               n_outside_cohort=len(labels)-n,n_classes=labels.nunique()))
    missing = pd.DataFrame(cohort_rows)
    overview.to_csv(output_dir/'data_overview.csv',index=False)
    counts.to_csv(output_dir/'class_counts.csv',index=False)
    missing.to_csv(output_dir/'task_cohorts.csv',index=False)
    clinical.to_parquet(output_dir/'paired_clinical.parquet')
    effects, moments, quality = [], [], []
    for modality,filename in FILES.items():
        print('Analyzing',modality,'over all paired samples/features',flush=True)
        table = feature_variance_by_task(data_root/filename,modality,clinical,groups,
            patient_summary_path=output_dir/f'{modality}_patient_summary.csv')
        quality.append(table.attrs['value_quality'])
        table.to_parquet(output_dir/f'{modality}_feature_variance.parquet',index=False)
        summary,selected = class_moments(data_root/filename,modality,clinical,groups,table,n_features)
        summary.to_csv(output_dir/f'{modality}_class_moments.csv',index=False)
        selected.to_parquet(output_dir/f'{modality}_selected_features.parquet')
        effects.append(table);moments.append(summary)
    pd.DataFrame(quality).to_csv(output_dir/'value_quality.csv',index=False)
    effects = pd.concat(effects,ignore_index=True)
    pd.concat(moments,ignore_index=True).to_csv(output_dir/'class_moments.csv',index=False)
    top = effects.sort_values('eta_squared',ascending=False).groupby(['task','modality']).head(12)
    top.to_csv(output_dir/'top_features_by_task.csv',index=False)
    (output_dir/'provenance.json').write_text(json.dumps(dict(data_root=str(data_root.resolve()),
        paired_samples=len(clinical),clinical_known=int(clinical.CANCER_TYPE_ACRONYM.notna().sum()),
        rna_transform='none',methylation_transform='none',variance_ddof=0,class_moment_variance_ddof=1,
        eta_squared='Patient-weighted between-class variance / total variance; labels missing per task excluded',
        raw_inputs=True,autoencodix_feature_filter=False,train_test_evaluation=False,
        feature_selection='Descriptive only; no significance or predictive claim'),indent=2)+'\n')
    return effects


def plot_variance_association(effects, tasks, output_dir):
    """Contrast large variance with label association; each modality has its own scale."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(tasks),2,figsize=(13,3.5*len(tasks)),squeeze=False)
    for row,task in enumerate(tasks):
        for col,modality in enumerate(FILES):
            group=effects.loc[effects.task.eq(task)&effects.modality.eq(modality)]
            group=group.loc[group.total_variance.gt(0)&group.eta_squared.notna()]
            ax=axes[row,col]
            ax.scatter(group.total_variance,group.eta_squared,s=3,alpha=.2,rasterized=True)
            for item in group.nlargest(3,'eta_squared').itertuples():
                ax.annotate(item.feature,(item.total_variance,item.eta_squared),fontsize=7)
            ax.set(xscale='log',ylim=(-.02,1.02),xlabel='Total feature variance',ylabel='Between-class fraction eta²',title=f'{task} · {modality}')
            ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(Path(output_dir)/f'variance_vs_class_association.{suffix}',dpi=160)
    plt.close(fig)


def plot_task_boxplots(task, effects, clinical, output_dir):
    """Plot the strongest descriptive eta² feature of each modality, all classes."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    labels=task_groups(clinical)[task]
    order=labels.value_counts().index.tolist()
    fig,axes=plt.subplots(1,2,figsize=(14,max(4,.3*len(order))),squeeze=False)
    for ax,modality in zip(axes.flat,FILES):
        table=effects.loc[effects.task.eq(task)&effects.modality.eq(modality)].dropna(subset=['eta_squared'])
        if table.empty:continue
        best=table.nlargest(1,'eta_squared').iloc[0]
        data=pd.read_parquet(Path(output_dir)/f'{modality}_selected_features.parquet',columns=[best.feature])
        frame=pd.DataFrame({'group':labels,'value':data[best.feature]}).dropna()
        sns.boxplot(data=frame,x='value',y='group',order=order,showfliers=False,color='#77aadd',ax=ax)
        ax.set(title=f'{modality} · gene ID {best.feature} · eta²={best.eta_squared:.3f}',
               xlabel='Stored RNA value' if modality=='RNA' else 'Stored methylation value',ylabel='Class')
    fig.suptitle(f'{task}: illustrative features, selected on these same data')
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ('png','pdf'):fig.savefig(Path(output_dir)/f'{task}_feature_boxplots.{suffix}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/raw/TCGAPanCancerAutoencodix'))
    parser.add_argument('--output-dir',type=Path,default=Path('beta_vae_experiments/analysis_results/tcga_data'))
    args=parser.parse_args()
    result=run_tcga_data_analysis(args.data_root,args.output_dir)
    clinical=pd.read_parquet(args.output_dir/'paired_clinical.parquet')
    plot_variance_association(result,['cancer_type','sex_all','brca_subtype','ucec_subtype'],args.output_dir)
    for task in task_groups(clinical):plot_task_boxplots(task,result,clinical,args.output_dir)
