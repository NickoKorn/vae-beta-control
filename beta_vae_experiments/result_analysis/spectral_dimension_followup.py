"""Phase 3/4 follow-up on the same saved attribution/probe results, no new graph."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from beta_vae_experiments.result_analysis.spectral_dimension_analysis import DEFAULT_OUT,safe_rho
from .utils import DEFAULT_SWEEP, posterior
from beta_vae_experiments.metrics.complementary.graph_signals.dimension_attribution import align_rows,match_dimensions


def run(out=DEFAULT_OUT,sweep=DEFAULT_SWEEP):
    out,sweep=Path(out),Path(sweep)
    profiles=pd.read_parquet(out/'spectral_dimension_profiles.parquet')
    importance=pd.read_parquet(out/'dimension_task_importance.parquet')
    states=pd.read_parquet(out/'spectral_run_summary.parquet')
    mu=profiles[profiles.representation.eq('mu')]
    singles=importance[importance.representation.eq('mu') & importance.neuron_id.ge(0)]
    ids=np.load(out/'reference_basis.npz')['sample_ids'].astype(str)
    # Group interactions: ΔAUC(group) - sum ΔAUC(single), not additive by assumption.
    interactions=[]
    for (run,rep),g in importance.groupby(['run_name','representation']):
        individual=g[g.neuron_id.ge(0)].set_index('neuron_id')
        for r in g[g.neuron_id.lt(0)].itertuples():
            dims=[int(x) for x in r.dimensions.split(',')]
            for kind in ['fixed','retrained']:
                interactions.append(dict(run_name=run,representation=rep,group=r.group,probe=kind,
                    group_loss=getattr(r,f'delta_auc_{kind}'),sum_individual_loss=individual.loc[dims,f'delta_auc_{kind}'].sum(),
                    nonadditivity=getattr(r,f'delta_auc_{kind}')-individual.loc[dims,f'delta_auc_{kind}'].sum()))
    pd.DataFrame(interactions).to_parquet(out/'dimension_group_interactions.parquet',index=False)
    # Neuron-wise KL, variance, spectral centroid and task relevance changes.
    attrs=mu.drop_duplicates(['run_name','neuron_id'])[['run_name','neuron_id','trajectory_id','seed','beta_per_dim','reference_neuron','alignment_abs_r','KL_j','var_mu_j','spectral_centroid','spectral_entropy']]
    attrs=attrs.merge(singles[['run_name','neuron_id','delta_auc_fixed','delta_auc_retrained','AUC_full']],on=['run_name','neuron_id'])
    dynamics=[];lag=[]
    for (traj,j),g in attrs[attrs.reference_neuron.ge(0) & attrs.beta_per_dim.gt(0)].groupby(['trajectory_id','reference_neuron']):
        g=g.sort_values('beta_per_dim');cols=['KL_j','var_mu_j','spectral_centroid','spectral_entropy','delta_auc_fixed','delta_auc_retrained']
        if len(g)<2:continue
        dt=np.diff(np.log(g.beta_per_dim));loss=-np.diff(g.AUC_full)/dt
        # Do not bridge a missing beta state after an ambiguous/collapsed match.
        all_beta=np.sort(states[states.trajectory_id.eq(traj)&states.beta_per_dim.gt(0)].beta_per_dim.unique())
        position=np.searchsorted(all_beta,g.beta_per_dim);adjacent=np.diff(position)==1
        change={c:np.where(adjacent,np.diff(g[c])/dt,np.nan) for c in cols}
        for i in range(len(dt)):
            dynamics.append(dict(trajectory_id=traj,seed=int(g.seed.iloc[0]),reference_neuron=j,beta_start=g.beta_per_dim.iloc[i],beta_end=g.beta_per_dim.iloc[i+1],
                                 adjacent_beta=bool(adjacent[i]),alignment_abs_r=float(min(g.alignment_abs_r.iloc[i],g.alignment_abs_r.iloc[i+1])),
                                 auc_loss_per_log_beta=loss[i] if adjacent[i] else np.nan,**{f'change_{c}':v[i] for c,v in change.items()}))
        for c,x in change.items():
            for k in range(4):
                # Lag k means exactly k positions of the original beta grid.
                y=np.where(adjacent,loss,np.nan)
                if not adjacent.all() and k>0:continue
                lag.append(dict(trajectory_id=traj,reference_neuron=j,feature=c,lag=k,n_pairs=max(0,len(x)-k),rho=safe_rho(x[:-k] if k else x,y[k:] if k else y)))
    pd.DataFrame(dynamics).to_parquet(out/'neuron_beta_dynamics.parquet',index=False)
    pd.DataFrame(lag).to_parquet(out/'neuron_kl_task_lag_correlations.parquet',index=False)
    # Fingerprint change measures separate spectral redistribution from scale.
    from scipy.spatial.distance import jensenshannon
    changes=[]
    for (traj,j),g in mu[mu.reference_neuron.ge(0)].groupby(['trajectory_id','reference_neuron']):
        f=g.pivot(index='beta_per_dim',columns='band',values='energy_fraction').sort_index()
        e=g.pivot(index='beta_per_dim',columns='band',values='energy_abs').reindex(f.index)
        for k in range(1,len(f)):
            a,b=f.iloc[k-1].to_numpy(),f.iloc[k].to_numpy()
            valid=np.isfinite(a).all() and np.isfinite(b).all() and a.sum()>0 and b.sum()>0
            changes.append(dict(trajectory_id=traj,reference_neuron=j,beta_start=f.index[k-1],beta_end=f.index[k],
                                fingerprint_L1=float(abs(a-b).sum()) if valid else np.nan,
                                fingerprint_JS_distance=float(jensenshannon(a,b)) if valid else np.nan,
                                absolute_energy_change=float(e.iloc[k].sum()-e.iloc[k-1].sum())))
    pd.DataFrame(changes).to_parquet(out/'neuron_fingerprint_changes.parquet',index=False)

    # Across-seed descriptive activation matching, same d and beta. Raw axis
    # index is never averaged as if it had a universal semantic identity.
    matching=[]
    meta=attrs.drop_duplicates('run_name')
    for (d,beta),g in importance.drop_duplicates('run_name').groupby(['latent_dim','beta_per_dim']):
        ref=g.sort_values('seed').iloc[0]
        mr,_,ir=posterior(sweep/'run_analysis'/ref.run_name/'posteriors','test');mr=align_rows(mr,ir,ids)
        for r in g.itertuples():
            m,_,i=posterior(sweep/'run_analysis'/r.run_name/'posteriors','test');m=align_rows(m,i,ids);idx,q=match_dimensions(mr,m)
            for j in range(d):matching.append(dict(latent_dim=d,beta_per_dim=beta,seed=r.seed,reference_seed=int(ref.seed),run_name=r.run_name,neuron_id=j,matched_reference_neuron=int(idx[j]),alignment_abs_r=q[j],transductive=True))
    matched=pd.DataFrame(matching);matched.to_parquet(out/'cross_seed_activation_alignment.parquet',index=False)
    # Online-available threshold alarms on held-out predictions, separate from
    # the retrospectively 'lasting' descriptive alarm definition.
    predictions=pd.read_parquet(out/'held_out_predictions.parquet')
    dynamics=pd.read_parquet(out/'spectral_beta_dynamics.parquet')
    predictions=predictions.merge(dynamics[['run_name','AUC','beta_per_dim']],on='run_name',validate='many_to_one')
    predictions['event']=predictions.observed>.02*predictions.AUC
    predictions['alarm']=predictions.predicted>.02*predictions.AUC
    scores=[]
    for (name,seed),g in predictions.groupby(['model','held_seed']):
        negative=~g.event;positive=g.event
        scores.append(dict(model=name,held_seed=seed,n=len(g),n_events=int(positive.sum()),n_nonevents=int(negative.sum()),
                           false_alarm_rate=float(g.loc[negative,'alarm'].mean()) if negative.any() else np.nan,
                           sensitivity=float(g.loc[positive,'alarm'].mean()) if positive.any() else np.nan,
                           precision=float(g.loc[g.alarm,'event'].mean()) if g.alarm.any() else np.nan))
    alarms=pd.DataFrame(scores);alarms.to_csv(out/'held_out_alarm_scores.csv',index=False)
    # Robustness of across-neuron associations, avoid pooled beta-point p-values.
    aggregate=[]
    for (d,beta),g in matched[matched.seed.ne(matched.reference_seed)].groupby(['latent_dim','beta_per_dim']):
        aggregate.append(dict(latent_dim=d,beta_per_dim=beta,median_alignment=g.alignment_abs_r.median(),fraction_alignment_ge_08=(g.alignment_abs_r>=.8).mean()))
    robust=pd.DataFrame(aggregate);robust.to_csv(out/'cross_seed_alignment_summary.csv',index=False)
    # Mark temporal redistribution descriptively in the matched beta=0 axes.
    migration=[]
    for traj,g in attrs[attrs.reference_neuron.ge(0)].groupby('trajectory_id'):
        t=g.pivot(index='beta_per_dim',columns='reference_neuron',values='delta_auc_fixed').sort_index()
        for i in range(1,len(t)):
            common=t.iloc[i].notna()&t.iloc[i-1].notna()
            v=t.iloc[i][common]-t.iloc[i-1][common]
            migration.append(dict(trajectory_id=traj,beta_start=t.index[i-1],beta_end=t.index[i],task_loss_in_some=bool((v<-.01).any()),task_gain_in_others=bool((v>.01).any()),
                                  redistribution_candidate=bool((v<-.01).any() and (v>.01).any()),n_common=int(common.sum())))
    pd.DataFrame(migration).to_parquet(out/'task_redistribution_candidates.parquet',index=False)
    # Annotate time-series figures at the SAME trajectory's AUC loss point.
    import matplotlib.pyplot as plt
    bands=mu.band.unique();example='d8_seed13';sub=mu[mu.trajectory_id.eq(example)&mu.reference_neuron.ge(0)]
    probe=singles[singles.trajectory_id.eq(example)].groupby('beta_per_dim').AUC_full.first().sort_index()
    hits=probe[probe<.98*probe.iloc[0]];collapse=hits.index[0] if len(hits) else np.nan
    for band in bands:
        fig,ax=plt.subplots(figsize=(8,4))
        for j,g in sub[sub.band.eq(band)].groupby('reference_neuron'):
            g=g.sort_values('beta_per_dim');ax.plot(g.beta_per_dim,g.energy_abs,marker='.',label=f'matched ref {j}')
        if np.isfinite(collapse):ax.axvline(collapse,color='red',linestyle='--',label='validation AUC relative loss >2%')
        ax.set_xscale('symlog',linthresh=5e-6);ax.set_xlabel('β/d');ax.set_ylabel('absolute centred μ energy');ax.set_title(f'{example}, {band}: descriptive matched activation profiles');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'figures'/f'matched_neuron_energy_{band}.png',dpi=150);plt.close(fig)
    # Mu versus fresh z: sum within each model first, then aggregate models.
    comparison=profiles[profiles.latent_dim.eq(8)].groupby(['run_name','beta_per_dim','representation','band']).energy_abs.sum().reset_index()
    totals=comparison.groupby(['run_name','representation']).energy_abs.transform('sum')
    comparison['share']=comparison.energy_abs/totals.where(totals>1e-24)
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for ax,rep in zip(axs,['mu','z_redrawn']):
        g=comparison[comparison.representation.eq(rep)].pivot_table(index='beta_per_dim',columns='band',values='share',aggfunc='median')
        im=ax.imshow(g,aspect='auto',vmin=0,vmax=1);ax.set_yticks(range(len(g)),[f'{b:g}' for b in g.index]);ax.set_xticks(range(len(g.columns)),g.columns,rotation=45);ax.set_title(rep+' · d=8, median over seeds');ax.set_ylabel('β/d')
    fig.colorbar(im,ax=list(axs),label='model-total energy share');fig.savefig(out/'figures'/'mu_vs_z_band_profiles.png',dpi=150,bbox_inches='tight');plt.close(fig)
    comparison.to_parquet(out/'mu_z_comparison.parquet',index=False)
    # Same forecasting implementation, now targeting the actual sampled-z
    # probe performance; mu remains the spectral representation.
    from beta_vae_experiments.result_analysis.spectral_dimension_analysis import phase3
    zout=out/'z_forecasting';zout.mkdir(exist_ok=True)
    ridge=pd.read_parquet(out/'band_reconstructability.parquet')
    _,zev=phase3(states,importance,ridge,profiles,zout,target_representation='z_redrawn')
    zmean=zev.groupby('model').rmse.mean().to_frame('mean_held_seed_RMSE')
    zpaired=zev.pivot(index='held_seed',columns='model',values='rmse')
    zdelta=(zpaired.beta_auc_standard_spectral-zpaired.beta_auc_standard).dropna().to_numpy()
    zboot=np.random.default_rng(31).choice(zdelta,(2000,len(zdelta)),replace=True).mean(1)

    per_neuron=mu.drop_duplicates(['run_name','neuron_id']).set_index(['run_name','neuron_id'])
    totals=mu.groupby(['run_name','neuron_id']).energy_abs.sum()
    per_neuron=per_neuron.join(totals.rename('total_energy'))
    shifts=[]
    for run,g in per_neuron.groupby(level=0):
        good=g.total_energy.gt(1e-24) & g.spectral_centroid.notna()
        weighted=np.average(g.loc[good,'spectral_centroid'],weights=g.loc[good,'total_energy']) if good.any() else np.nan
        shifts.append(dict(run_name=run,beta_per_dim=g.beta_per_dim.iloc[0],latent_dim=g.latent_dim.iloc[0],energy_weighted_centroid=weighted,
                           median_neuron_band_entropy=g.loc[good,'spectral_entropy'].median(),n_nonconstant=int(good.sum())))
    shifts=pd.DataFrame(shifts);shifts.to_parquet(out/'spectral_specialization_summary.parquet',index=False)
    beta_summary=shifts.groupby('beta_per_dim').agg(median_centroid=('energy_weighted_centroid','median'),median_entropy=('median_neuron_band_entropy','median'),defined_models=('energy_weighted_centroid','count'))
    max_share=mu.groupby(['run_name','neuron_id']).energy_fraction.max().dropna()

    text=['## Ergänzende Phase-3/4-Prüfungen',
          f'A: Der Median des größten Bandanteils eines nichtkonstanten Neurons beträgt {max_share.median():.3f}; der Anteil mit mehr als 50% Energie in einem Band beträgt {(max_share>.5).mean():.1%}. Bandgrößen sind ungleich; Konzentration allein beweist keine Aufgabenrelevanz.',
          'B: Energiegewichteter Schwerpunkt von μ pro Modell, danach Median über Modelle. Nullenergie-Modelle bleiben undefiniert; die Tabelle ist deskriptiv und behauptet keinen garantierten Tiefpass.',
          beta_summary.to_markdown(),
          'Neuronen-Dynamik enthält KL, Varianz, Bandprofil und fixe/neu trainierte Ablationsrelevanz. Matching wird nicht als Nachweis biologischer Identität behandelt; fehlende Zwischenstufen werden nicht als benachbarte β-Schritte verbunden.',
          'Gruppenablationen: Nichtadditivität = Gruppenverlust minus Summe der Einzelverluste. Vorzeichen beschreibt Abweichung von Additivität, keine eindeutige kausale Synergie.',
          'Cross-Seed-Matching: absolute μ-Aktivierungskorrelation auf gleichen Testpatienten; deskriptiv. Schwache Matches sprechen gegen feste neuron_id-basierte Steuerung.',
          robust.groupby('latent_dim').median(numeric_only=True).to_markdown(),
          'Online verfügbare Alarme auf zurückgehaltenen Seed-Gruppen: prognostizierter nächster AUC-Verlust > 2% der aktuellen AUC. Die Schwelle wurde nicht anhand der zurückgehaltenen Gruppe gewählt. Das beschreibt einen Schritt Vorlauf; Lags 0–3 werden separat als Korrelationen berichtet.',
          alarms.to_markdown(index=False),
          'Zusätzliche Vorhersageprüfung: spektrales μ → zukünftige Valid-AUC auf drei neu gezogenen z. Gleiche zurückgehaltene Seed-Gruppen und identische Baselines.',
          zmean.to_markdown(),
          f'z-AUC: RMSE-Differenz spektral minus Standardbaseline {zdelta.mean():+.5f}, exploratives 95%-Seed-Bootstrap-Intervall [{np.quantile(zboot,.025):+.5f}, {np.quantile(zboot,.975):+.5f}].',
          'Informationsverlagerung: zugleich steigende/sinkende Einzelablationseffekte sind Kandidaten, keine bewiesene Umlagerung. Independentes Training, redundante Signale und Probe-Verhalten können dasselbe Muster erzeugen.',
          'Controller-Entscheidung: vorerst keine empirisch abgesicherte Empfehlung für automatische β-Anpassung. Selbst günstige Vorhersagewerte benötigen unabhängige Kalibrierung und Replikation; Testdaten haben bereits die deskriptive Frequenzanalyse informiert.']
    p=out/'spectral_control_findings.md';original=p.read_text().split('\n\n## Ergänzende Phase-3/4-Prüfungen')[0];p.write_text(original+'\n\n'+'\n\n'.join(text))
    print('Follow-up complete:',out)

if __name__=='__main__':run()
