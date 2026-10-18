"""Read-only geometry evaluation of saved posteriors/checkpoints. Never trains.

Uses the existing spectral study's input cache, posterior reader and ID alignment,
and the existing exploratory segmented breakpoint estimator. No decoder is called.
"""
from beta_vae_experiments.user_interface.config import load_config
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from .utils import DEFAULT_SWEEP, DEFAULT_X, posterior, digest, breakpoint, correlation
from ..metrics.complementary.graph_signals.dimension_attribution import align_rows
from ..metrics.complementary.curvature.encoder_geometry import EncoderGeometryConfig, run_encoder_geometry_metrics_pipeline

BASELINES = ['posterior_conditional_kl_total_mean', 'posterior_active_units_count_eps_0p01',
             'dimensionality_mu_effective_rank', 'posterior_collapsed_dimensions_fraction',
             'geometry_mu_lcmc_zadu_k20', 'geometry_mu_trustworthiness_zadu_k20', 'recon_test_mse',
             'downstream_CANCER_TYPE_ACRONYM_z_test_roc_auc_ovo']
AUC=BASELINES[-1]


def load_model(sweep, run, input_dim):
    """Reconstruct the native architecture only when a strict final state exists."""
    import torch
    path=sweep/'run_analysis'/run.run_name/'model_final.pt'
    if not path.exists():
        return None
    from ..new_beta_varix_experiments import build_varix_config
    from ..data_pipeline import build_data_config
    from autoencodix.modeling._varix_architecture import VarixArchitecture
    settings=json.loads((sweep/'experiment_config.json').read_text())
    data=settings['dataset']
    config=build_varix_config(build_data_config(Path(__file__).resolve().parents[2],data['id'],data.get('options')),
                             data['id'],dict(settings['training'],device='cpu'),run._asdict(),global_seed=run.seed)
    with torch.random.fork_rng():
        model=VarixArchitecture(config,input_dim)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model


def evaluate(plan_path):
    root=Path(__file__).resolve().parents[2]
    plan=load_config(plan_path)
    def resolve(value):
        path=Path(value)
        return path if path.is_absolute() else root/path
    sweep=resolve(plan['sweep']);xfile=resolve(plan['input_splits']);output=resolve(plan['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    config=EncoderGeometryConfig(enabled=True,**plan.get('geometry',{}))
    with np.load(xfile,allow_pickle=False) as saved:
        x,ids=saved['x_test'],saved['ids_test'].astype(str)
    if len(set(ids))!=len(ids) or len(ids)!=len(x):
        raise ValueError('Invalid original input IDs')
    overview=pd.read_csv(sweep/'experiment_metrics.csv')
    overview=overview[overview.status.eq('ok')].copy()
    if plan.get('latent_dims'):
        overview=overview[overview.latent_dim.isin(plan['latent_dims'])]
    if plan.get('seeds'):
        overview=overview[overview.seed.isin(plan['seeds'])]
    input_hash=digest(xfile)
    inventory, all_metrics=[],[]
    for count,run in enumerate(overview.itertuples(index=False),1):
        folder=sweep/'run_analysis'/run.run_name
        mu,lv,posterior_ids=posterior(folder/'posteriors','test')
        mu=align_rows(mu,posterior_ids,ids);lv=align_rows(lv,posterior_ids,ids)
        inputs=SimpleNamespace(run_name=run.run_name,x_test=x,mu_test=mu,logvar_test=lv,
                               sample_ids_test=ids,metadata=run._asdict())
        directory=output/'metrics_artifacts'/run.run_name/'encoder_geometry'
        model_path=folder/'model_final.pt'
        signature=dict(input_sha256=input_hash,posterior_sha256=digest(folder/'posteriors/posterior_test.npz'),
                       model_sha256=digest(model_path) if model_path.exists() else None,config=vars(config),
                       metric_code_sha256=digest(Path(__file__).resolve().parents[1]/'metrics/complementary/curvature/encoder_geometry.py'),version=2)
        cache=directory/'cache_signature.json'
        # JSON turns tuples into lists; compare the serialized canonical forms.
        canonical=json.loads(json.dumps(signature))
        if cache.exists() and json.loads(cache.read_text())==canonical and (directory/'metrics.parquet').exists():
            metrics=pd.read_parquet(directory/'metrics.parquet')
        else:
            model=load_model(sweep,run,x.shape[1])
            metrics=run_encoder_geometry_metrics_pipeline(inputs,np.arange(len(x)),x,mu,config,output,model)
            cache.write_text(json.dumps(signature,indent=2)+'\n')
        metrics=metrics.assign(run_name=run.run_name,beta=run.beta,latent_dim=run.latent_dim,seed=run.seed)
        all_metrics.append(metrics)
        inventory.append(dict(run_name=run.run_name,input_shape=list(x.shape),posterior_shape=list(mu.shape),
                              checkpoint_available=model_path.exists(),decoder_used=False))
        if count%20==0 or count==len(overview):
            print(f'Geometry {count}/{len(overview)}',flush=True)
    if not all_metrics:
        raise ValueError('No matching completed runs')
    long=pd.concat(all_metrics,ignore_index=True)
    long.to_parquet(output/'encoder_latent_geometry.parquet',index=False)
    pd.DataFrame(inventory).to_parquet(output/'artifact_inventory.parquet',index=False)
    wide=long.pivot(index='run_name',columns='metric_name',values='value')
    combined=overview.merge(wide,on='run_name',validate='one_to_one')
    combined.to_parquet(output/'geometry_beta_combined.parquet',index=False)
    correlation_rows,knees=[],[]
    metrics=list(wide.columns)
    for (dimension,seed),group in combined.groupby(['latent_dim','seed']):
        group=group.sort_values('beta')
        for name in metrics:
            for target in [AUC,*BASELINES[:1]]:
                if target not in group:
                    continue
                correlation_rows.append(dict(latent_dim=dimension,seed=seed,metric=name,target=target,
                                              n_pairs=int(group[[name,target]].dropna().shape[0]),
                                              spearman=correlation(group[name],group[target]),
                                              pearson=correlation(group[name],group[target],'pearson')))
        for name in metrics+[c for c in BASELINES if c in group]:
            beta,gain=breakpoint(group.beta.to_numpy(),group[name].to_numpy())
            knees.append(dict(latent_dim=dimension,seed=seed,metric=name,beta_knee=beta,
                              fit_gain=gain,n_observations=int(group[name].notna().sum()),
                              exploratory=True))
    pd.DataFrame(correlation_rows).to_parquet(output/'within_trajectory_correlations.parquet',index=False)
    knee_table=pd.DataFrame(knees)
    knee_table.to_parquet(output/'exploratory_change_points.parquet',index=False)
    compare_change_points(knee_table,output)
    plot_curves(combined,output)
    missing=sum(not item['checkpoint_available'] for item in inventory)
    (output/'findings.md').write_text(f'''# Encoder und Latentraum: Auswertung ohne Decoder

{len(overview)} gespeicherte Modelle, {len(x)} gepaarte Testproben mit {x.shape[1]} Merkmalen.
Pro Modell höchstens {config.max_samples} identische, seed-fixierte Proben für die Geometrie.
K-Werte: {list(config.k_values)}. Kein Training, keine Decoder-Auswertung.

{missing}/{len(overview)} finale Checkpoints fehlen am geprüften Standardpfad.
Für diese Modelle sind Encoder-Jacobians nicht verfügbar; gespeicherte μ/logvar
ersetzen keine Ableitungen. Der Status bleibt missing_checkpoint.

Die Krümmung einer glatten Inputmannigfaltigkeit ist ohne verifizierte glatte
Karte und Rangbedingungen nicht definiert. Die Datei enthält dafür keinen
numerischen Ersatzwert. Graph-Krümmung aus der vorhandenen Pipeline bleibt eine
andere Größe. KNN-Pfadlängen sind Stichproben-Graphdistanzen, keine exakten Geodäten.

geometry_beta_combined.parquet verknüpft neue Geometrie mit vorhandener Rate,
Active Units, Effective Rank, Collapse, Preservation, Reconstruction und Cancer-AUC.
Neue μ-Metriken werden mit der vorhandenen z-Probe verglichen; sie sind nicht
dieselbe Repräsentation. Die Preservation-Baselines verwendeten ihren ursprünglichen
Auswertungsausschnitt; k ist bei verschiedenen Stichprobengrößen nicht dieselbe Skala.

Korrelationen werden je Seed und Dimension berechnet. Change Points verwenden den
bestehenden segmentierten Fit und sind explorativ; selbst ein früherer Knick ist
kein validiertes Frühwarnsignal. Wenige β-Punkte, Auswahl am selben Sweep und
fehlende zurückgehaltene Bestätigung erlauben keine Controller-Empfehlung.

Ob geometrische Merkmale zusätzlichen Nutzen über Rate und AUC liefern, muss mit
zurückgehaltenen Trajektorien bestätigt werden. Dafür wurde hier kein Controller
implementiert. Numerische Definitionen und Quellen: metrics/complementary/curvature/encoder_geometry.md.
''')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return combined


def compare_change_points(knees,output):
    """Report seed-level ordering and sensitivity; never label it confirmed lead."""
    auc=knees[knees.metric.eq(AUC)][['latent_dim','seed','beta_knee','fit_gain']].rename(
        columns={'beta_knee':'auc_beta_knee','fit_gain':'auc_fit_gain'})
    comparison=knees[~knees.metric.eq(AUC)].merge(auc,on=['latent_dim','seed'],validate='many_to_one')
    rows=[]
    for threshold in (.1,.25,.5):
        part=comparison.copy()
        part['fit_gain_threshold']=threshold
        part['both_fits_eligible']=(part.fit_gain.ge(threshold)&part.auc_fit_gain.ge(threshold)
                                   &part.beta_knee.gt(0)&part.auc_beta_knee.gt(0))
        part['earlier_than_auc']=part.both_fits_eligible & part.beta_knee.lt(part.auc_beta_knee)
        rows.append(part)
    comparison=pd.concat(rows,ignore_index=True)
    comparison.to_parquet(output/'change_point_ordering_by_seed.parquet',index=False)
    summary=comparison.groupby(['latent_dim','metric','fit_gain_threshold']).agg(
        n_seeds=('seed','nunique'),n_eligible=('both_fits_eligible','sum'),
        n_earlier_than_auc=('earlier_than_auc','sum'))
    summary.to_csv(output/'change_point_ordering_summary.csv')
    chosen=['mu_norm_mean','mu_knn_path_to_chord_median_k20','mu_delta_over_diameter',
            'posterior_conditional_kl_total_mean','geometry_mu_trustworthiness_zadu_k20']
    table=summary.reset_index()
    table=table[table.fit_gain_threshold.eq(.25)&table.metric.isin(chosen)].groupby('metric')[
        ['n_seeds','n_eligible','n_earlier_than_auc']].sum()
    lines=['# Explorative Knickpunkte gegenüber Cancer-AUC', '',
           'Fit-Gain-Schwelle 0.25, dieselben gespeicherten Trajektorien, keine gehaltene Testauswertung.',
           'n_seeds unten summiert Seeds über Dimensionen: tatsächlich Anzahl Trajektorien, nicht unabhängiger Datensätze.', '',
           '| Metrik | Trajektorien | Beide Fits geeignet | Früher als AUC |',
           '|---|---:|---:|---:|']
    for metric,row in table.iterrows():
        lines.append(f"| {metric} | {int(row.n_seeds)} | {int(row.n_eligible)} | {int(row.n_earlier_than_auc)} |")
    lines += ['', 'Die Rate selbst zeigt frühe Knicke. Ein früher Geometrie-Knick allein belegt deshalb',
              'keinen zusätzlichen Vorhersagenutzen. KNN-Pfadlängen hängen von Stichprobe und k ab;',
              'Hyperbolizität ist keine Sectional Curvature. Weitere Schwellen 0.1 und 0.5 stehen in der CSV.',
              'Controller-Eignung oder eine stabile Frühwarnschwelle ist damit nicht bestätigt.']
    (output/'change_point_summary.md').write_text('\n'.join(lines)+'\n')



def plot_curves(frame,output):
    import matplotlib.pyplot as plt
    names=['mu_norm_mean','mu_to_input_distance_ratio_median','mu_delta_over_diameter',
           'mu_knn_path_to_chord_median_k20',*BASELINES[:2],BASELINES[5],AUC]
    names=[name for name in names if name in frame]
    figure,axes=plt.subplots(2,4,figsize=(17,8),squeeze=False)
    for axis,name in zip(axes.flat,names):
        for dimension,group in frame.groupby('latent_dim'):
            summary=group.groupby('beta')[name].agg(['median',lambda v:v.quantile(.25),lambda v:v.quantile(.75)])
            axis.plot(summary.index/dimension,summary['median'],label=f'd={dimension}')
            axis.fill_between(summary.index/dimension,summary.iloc[:,1],summary.iloc[:,2],alpha=.12)
        axis.set_title(name,fontsize=8);axis.set_xlabel('β/d');axis.set_xscale('symlog',linthresh=1e-6);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    figure.tight_layout();figure.savefig(output/'geometry_beta_overview.png',dpi=170);plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    evaluate(parser.parse_args().plan)
