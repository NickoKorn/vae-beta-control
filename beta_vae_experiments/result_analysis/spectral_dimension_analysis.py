"""Reproduce the notebook spectral attribution study; no VAE training/controller.

Run from repo: python -m beta_vae_experiments.result_analysis.spectral_dimension_analysis
Inputs are explicit, hashed and aligned by IDs. Existing completed stages can
be reused only with matching provenance. Default probe evaluation is validation;
the graph and modal regressions describe the existing TEST reference graph.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from beta_vae_experiments.metrics.graph_signals import graph_spectrum as gs
from beta_vae_experiments.metrics.graph_signals.dimension_attribution import (
    align_rows,reference_bands,fingerprints,match_dimensions,probe_ablation,ridge_reconstruct)

from .utils import DEFAULT_SWEEP, DEFAULT_X, posterior, digest

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'beta_vae_experiments/results_26_09/spectral_dimension_analysis'
BASE_COLS={'rate':'posterior_conditional_kl_total_mean','active_units':'posterior_active_units_count_eps_0p01',
           'effective_rank':'dimensionality_mu_effective_rank','lcmc':'geometry_mu_lcmc_zadu_k20',
           'trustworthiness':'geometry_mu_trustworthiness_zadu_k20'}


def labels(clin,ids):
    y=clin.CANCER_TYPE_ACRONYM.reindex(ids)
    if y.isna().any():raise ValueError('Missing cancer labels; no fabricated labels')
    return y.astype(str).to_numpy()


def inventory(sweep,xfile,output):
    output.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(sweep/'experiment_metrics.csv');table=table[table.status.eq('ok')].copy()
    table['beta_per_dim']=(table.beta/table.latent_dim).round(10)
    table['trajectory_id']='d'+table.latent_dim.astype(str)+'_seed'+table.seed.astype(str)
    for key,col in BASE_COLS.items():
        table[key]=table[col] if col in table else np.nan
    with np.load(xfile,allow_pickle=False) as a:
        X=a['x_test'].astype(float);ids=a['ids_test'].astype(str)
    idx=gs.select_indices(len(X));X,ids=X[idx],ids[idx]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate input IDs')
    # Reference IDs must describe the same patient sets for all models.
    artifact_rows=[];hashes={}
    for r in table.itertuples():
        folder=sweep/'run_analysis'/r.run_name/'posteriors'
        sets=[]
        for split in ['train','valid','test']:
            p=folder/f'posterior_{split}.npz'
            if not p.exists():raise FileNotFoundError(f'Missing posterior: {p}')
            mu,lv,pi=posterior(folder,split)
            if mu.shape[1]!=r.latent_dim:raise ValueError('Run dimension disagrees with posterior')
            sets.append(set(pi));hashes[str(p.relative_to(sweep))]=digest(p)
            artifact_rows.append(dict(run_name=r.run_name,split=split,n_samples=len(mu),latent_dim=mu.shape[1],stored_z=False))
            if split=='test' and not set(ids).issubset(pi):raise ValueError('Input graph IDs do not match posterior test')
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('Overlapping train/valid/test IDs')
    shapes=pd.DataFrame(artifact_rows);shapes.to_parquet(output/'artifact_inventory.parquet',index=False)
    return table,X,ids,idx,hashes


def basis(X,ids,idx,output):
    W,L=gs.build_graph(X,k=20,laplacian='normalized',weights='binary')
    old=ROOT/'beta_vae_experiments/results_26_09/graph_spectrum_cache/graph_basis_tcga_split2_test_k20_normalized.npz'
    if old.exists():
        b=gs.load_basis(old);U,lam=b['U'],b['lam']
        if not np.array_equal(b['indices'],idx) or U.shape[0]!=len(ids):raise ValueError('Cached reference indices disagree')
        if not np.allclose(U.T@U,np.eye(U.shape[1]),atol=1e-8) or not np.allclose(L@U,U*lam,atol=1e-7):
            raise ValueError('Cached basis is inconsistent with actual input graph')
    else:U,lam=gs.fourier_basis(L)
    np.savez_compressed(output/'reference_basis.npz',U=U,lam=lam,indices=idx,sample_ids=ids)
    bands,names=reference_bands(lam)
    meta=[dict(band=name,n_modes=len(b),lambda_min=float(lam[b].min()),lambda_max=float(lam[b].max())) for b,name in zip(bands,names)]
    return U,lam,bands,names,meta


def phase1(table,U,lam,bands,names,ids,sweep,out,z_draws):
    rows=[];summaries=[];references={}
    for r in table.sort_values(['latent_dim','seed','beta']).itertuples():
        folder=sweep/'run_analysis'/r.run_name/'posteriors';mu,lv,pi=posterior(folder,'test')
        mu,lv=align_rows(mu,pi,ids),align_rows(lv,pi,ids)
        if r.trajectory_id not in references:
            if r.beta!=0:raise ValueError('Missing beta=0 reference model')
            ref=fingerprints(U,lam,mu,lv,bands,validate_basis=False);references[r.trajectory_id]=(mu,ref)
        ref_mu,ref=references[r.trajectory_id]
        match,quality=match_dimensions(ref_mu,mu)
        # Independently trained beta models: same neuron ID does NOT assert correspondence.
        mu_profile=fingerprints(U,lam,mu,lv,bands,validate_basis=False)
        for representation in ['mu','z_redrawn']:
            ps=[mu_profile] if representation=='mu' else [fingerprints(U,lam,mu+np.exp(.5*lv)*np.random.default_rng(31000+i).standard_normal(mu.shape),lv,bands,validate_basis=False) for i in range(z_draws)]
            E=np.mean([p['energy'] for p in ps],axis=0)
            total=np.mean([p['total'] for p in ps],axis=0)
            F=np.divide(E,total[:,None],out=np.full_like(E,np.nan),where=total[:,None]>1e-24)
            for j in range(r.latent_dim):
                for b,name in enumerate(names+['uncaptured']):
                    denominator=ref['energy'][match[j],b] if match[j]>=0 else np.nan
                    relative=E[j,b]/denominator if representation=='mu' and denominator>1e-24 else np.nan
                    rows.append(dict(run_name=r.run_name,trajectory_id=r.trajectory_id,seed=r.seed,beta=r.beta,beta_per_dim=r.beta_per_dim,latent_dim=r.latent_dim,
                                     neuron_id=j,representation=representation,band=name,energy_abs=E[j,b],energy_fraction=F[j,b],energy_relative=relative,
                                     reference_neuron=int(match[j]),alignment_abs_r=quality[j],KL_j=mu_profile['KL'][j],var_mu_j=mu_profile['variance'][j],active_unit=bool(mu_profile['active'][j]),
                                     spectral_centroid=np.nanmean([p['centroid'][j] for p in ps]) if total[j]>1e-24 else np.nan,
                                     spectral_entropy=np.nanmean([p['entropy'][j] for p in ps]) if total[j]>1e-24 else np.nan))
        e=mu_profile['energy'].sum(0);frac=e/e.sum() if e.sum()>1e-24 else np.full_like(e,np.nan)
        summaries.append(dict(run_name=r.run_name,trajectory_id=r.trajectory_id,seed=r.seed,beta=r.beta,beta_per_dim=r.beta_per_dim,latent_dim=r.latent_dim,
                              rate=r.rate,active_units=r.active_units,effective_rank=r.effective_rank,lcmc=r.lcmc,trustworthiness=r.trustworthiness,
                              **{f'energy_{n}':v for n,v in zip(names+['uncaptured'],e)},**{f'fraction_{n}':v for n,v in zip(names+['uncaptured'],frac)}))
    profiles=pd.DataFrame(rows);profiles.to_parquet(out/'spectral_dimension_profiles.parquet',index=False)
    states=pd.DataFrame(summaries);states.to_parquet(out/'spectral_run_summary.parquet',index=False)
    return profiles,states


def _phase2_sequential(table,profiles,U,bands,names,ids,sweep,clin,out,z_draws):
    rows=[];modes=[];bandscores=[]
    for number,r in enumerate(table.itertuples(),1):
        folder=sweep/'run_analysis'/r.run_name/'posteriors';mt,lt,it=posterior(folder,'train');mv,lv,iv=posterior(folder,'valid');me,le,ie=posterior(folder,'test')
        yt,yv=labels(clin,it),labels(clin,iv)
        mu=align_rows(me,ie,ids)
        groups=[(f'neuron_{j}',[j]) for j in range(r.latent_dim)]
        # Data-defined groups come from TRAIN variance, not validation/test task labels.
        ranked=np.argsort(-mt.var(0));half=max(1,r.latent_dim//2)
        groups += [('high_train_variance',ranked[:half]),('low_train_variance',ranked[half:])]
        if r.latent_dim>=4:groups.append(('top_two_train_variance',ranked[:2]))
        for rep in ['mu','z_redrawn']:
            draw_rows=[]
            for draw in range(1 if rep=='mu' else z_draws):
                rng=np.random.default_rng(41000+draw)
                tr=mt if rep=='mu' else mt+np.exp(.5*lt)*rng.standard_normal(mt.shape)
                ev=mv if rep=='mu' else mv+np.exp(.5*lv)*rng.standard_normal(mv.shape)
                for v in probe_ablation(tr,ev,yt,yv,groups):
                    draw_rows.append(dict(**v,draw=draw))
            df=pd.DataFrame(draw_rows)
            for (group,dimensions),g in df.groupby(['group','dimensions']):
                row={k:float(g[k].mean()) for k in ['AUC_full','AUC_fixed','delta_auc_fixed','AUC_retrained','delta_auc_retrained']}
                row.update(run_name=r.run_name,trajectory_id=r.trajectory_id,seed=r.seed,beta=r.beta,beta_per_dim=r.beta_per_dim,latent_dim=r.latent_dim,
                           representation=rep,evaluation_split='valid',group=group,dimensions=dimensions,neuron_id=int(group.split('_')[1]) if group.startswith('neuron_') else -1,
                           n_draws=len(g),draw_sd=float(g.delta_auc_fixed.std(ddof=1)) if len(g)>1 else np.nan)
                rows.append(row)
        per,bs=ridge_reconstruct(U,mu,bands)
        for k,v in enumerate(per):modes.append(dict(run_name=r.run_name,mode=k,r2=v,transductive=True))
        for name,v in zip(names,bs):bandscores.append(dict(run_name=r.run_name,band=name,r2=v,transductive=True))
        if number%10==0:print(f'Phase 2: {number}/{len(table)} runs',flush=True)
    importance=pd.DataFrame(rows);importance.to_parquet(out/'dimension_task_importance.parquet',index=False)
    pd.DataFrame(modes).to_parquet(out/'eigenmode_reconstructability.parquet',index=False)
    scores=pd.DataFrame(bandscores);scores.to_parquet(out/'band_reconstructability.parquet',index=False)
    return importance,scores



def _phase2_worker(row, U, bands, names, ids, sweep, clin, out, z_draws):
    folder=out/'run_cache'/str(row['run_name']);folder.mkdir(parents=True,exist_ok=True)
    expected=['dimension_task_importance.parquet','band_reconstructability.parquet','eigenmode_reconstructability.parquet']
    if not all((folder/name).exists() for name in expected):
        _phase2_sequential(pd.DataFrame([row]),None,U,bands,names,ids,sweep,clin,folder,z_draws)
    return tuple(pd.read_parquet(folder/name) for name in expected)


def phase2(table,profiles,U,bands,names,ids,sweep,clin,out,z_draws):
    from joblib import Parallel, delayed
    results=Parallel(n_jobs=4,verbose=10)(delayed(_phase2_worker)(row,U,bands,names,ids,sweep,clin,out,z_draws) for row in table.to_dict('records'))
    importance=pd.concat([v[0] for v in results],ignore_index=True)
    scores=pd.concat([v[1] for v in results],ignore_index=True)
    modes=pd.concat([v[2] for v in results],ignore_index=True)
    importance.to_parquet(out/'dimension_task_importance.parquet',index=False)
    scores.to_parquet(out/'band_reconstructability.parquet',index=False)
    modes.to_parquet(out/'eigenmode_reconstructability.parquet',index=False)
    return importance,scores


def safe_rho(a,b):
    a,b=np.asarray(a),np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    return float(spearmanr(a[ok],b[ok]).statistic) if ok.sum()>=4 and np.ptp(a[ok])>1e-14 and np.ptp(b[ok])>1e-14 else np.nan


def phase3(states,importance,scores,profiles,out,target_representation="mu"):
    # Primary dynamics use validation AUC on mu, matched to the spectral mu representation.
    auc=importance[importance.representation.eq(target_representation)].groupby('run_name').AUC_full.first()
    d=states.copy();d['AUC']=d.run_name.map(auc)
    b=scores.pivot(index='run_name',columns='band',values='r2').add_prefix('r2_')
    d=d.join(b,on='run_name')
    for rep in ['mu','z_redrawn']:
        a=importance[importance.representation.eq(rep)].groupby('run_name').AUC_full.first();d[f'AUC_{rep}']=d.run_name.map(a)
    # Axis-invariant per-dimension KL baseline, not merely the total rate.
    klrows=profiles[profiles.representation.eq('mu')].drop_duplicates(['run_name','neuron_id'])
    klstats=klrows.groupby('run_name').KL_j.agg(KL_max='max',KL_std='std')
    def kl_entropy(values):
        values=np.asarray(values,float);total=values.sum()
        if total<=1e-24:return np.nan
        q=values/total
        return float(-np.sum(q*np.log(np.maximum(q,1e-300)))/np.log(len(q)))
    klstats['KL_entropy']=klrows.groupby('run_name').KL_j.apply(kl_entropy)
    d=d.join(klstats,on='run_name')

    d['log_beta']=np.log(d.beta_per_dim.replace(0,np.nan));d['log_rate']=np.log(d.rate.clip(lower=1e-12))
    # Collapse-invariant aggregation; never treat neuron index across seeds as a common feature.
    cols=[c for c in d if c.startswith(('energy_','fraction_','r2_'))]+['rate','log_rate','KL_max','KL_std','KL_entropy','active_units','effective_rank','lcmc','trustworthiness']
    steps=[];cor=[];alerts=[]
    for traj,g in d[d.beta_per_dim.gt(0)].groupby('trajectory_id'):
        g=g.sort_values('beta_per_dim').reset_index(drop=True);dx=np.diff(g.log_beta)
        for col in cols:
            change=np.diff(g[col])/dx;loss=-np.diff(g.AUC)/dx
            for lag in range(4):
                x=change[:len(change)-lag] if lag else change;y=loss[lag:] if lag else loss
                cor.append(dict(trajectory_id=traj,seed=int(g.seed.iloc[0]),feature=col,lag=lag,n_pairs=int(np.isfinite(x*y).sum()),rho_values=safe_rho(g[col],g.AUC) if lag==0 else np.nan,
                                rho_change_vs_future_auc_loss=safe_rho(x,y)))
        baseline=g.AUC.iloc[0];hit=np.flatnonzero(g.AUC.to_numpy() < .98*baseline);auc_hit=int(hit[0]) if len(hit) else -1
        for col in ['rate','active_units','effective_rank','lcmc','trustworthiness']+[c for c in cols if c.startswith('energy_')]:
            v=g[col].to_numpy();start=v[0];loss=v < .95*start if np.isfinite(start) and start>0 else np.zeros(len(v),bool)
            lasting=[i for i in range(1,len(v)) if loss[i:].all()]
            alarm=lasting[0] if lasting else -1
            alerts.append(dict(trajectory_id=traj,seed=int(g.seed.iloc[0]),feature=col,alarm_step=alarm,auc_loss_step=auc_hit,
                               lead_steps=auc_hit-alarm if alarm>=0 and auc_hit>=0 else np.nan,false_alarm=bool(alarm>=0 and auc_hit<0),
                               threshold_energy_drop=.05,threshold_auc_relative_drop=.02,reference='first_positive_beta',retrospective=True))
        for i in range(1,len(g)-1):
            row=g.iloc[i].to_dict();row['target_future_auc_loss']=g.AUC.iloc[i]-g.AUC.iloc[i+1]
            row['past_auc_loss']=g.AUC.iloc[i-1]-g.AUC.iloc[i];row['log_beta_next']=g.log_beta.iloc[i+1]
            for col in cols:row[f'change_{col}']=(g[col].iloc[i]-g[col].iloc[i-1])/(g.log_beta.iloc[i]-g.log_beta.iloc[i-1])
            steps.append(row)
    step=pd.DataFrame(steps);step.to_parquet(out/'spectral_beta_dynamics.parquet',index=False)
    pd.DataFrame(cor).to_parquet(out/'trajectory_lag_correlations.parquet',index=False)
    pd.DataFrame(alerts).to_parquet(out/'retrospective_alerts.parquet',index=False)
    # Baseline includes beta AND its known next value (accounts for uneven sweep grid).
    base=['log_beta','log_beta_next','AUC','past_auc_loss','latent_dim']
    ordinary=['log_rate','KL_max','KL_std','KL_entropy','active_units','effective_rank','lcmc','trustworthiness']
    spectral=[c for c in cols if c.startswith(('fraction_','r2_')) and not c.endswith('B0') and not c.endswith('uncaptured')]
    models={'beta_past_auc':base,'beta_auc_standard':base+ordinary,
            'beta_auc_standard_spectral':base+ordinary+spectral}
    predictions=[]
    for name,features in models.items():
        # No fabricated values: complete cases shared across all models.
        common=list(dict.fromkeys(sum(models.values(),[])))+['target_future_auc_loss']
        good=step.dropna(subset=common).reset_index(drop=True)
        for train,test in GroupKFold(n_splits=len(good.seed.unique())).split(good,groups=good.seed):
            fit=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(good.loc[train,features],good.loc[train,'target_future_auc_loss'])
            pred=fit.predict(good.loc[test,features])
            for j,v in zip(test,pred):predictions.append(dict(model=name,run_name=good.run_name.iloc[j],trajectory_id=good.trajectory_id.iloc[j],held_seed=int(good.seed.iloc[j]),
                                                           observed=good.target_future_auc_loss.iloc[j],predicted=v))
    p=pd.DataFrame(predictions);p.to_parquet(out/'held_out_predictions.parquet',index=False)
    evaluation=[]
    for (name,seed),g in p.groupby(['model','held_seed']):evaluation.append(dict(model=name,held_seed=seed,n=len(g),rmse=np.sqrt(mean_squared_error(g.observed,g.predicted)),mae=np.mean(abs(g.observed-g.predicted)),rho=safe_rho(g.observed,g.predicted)))
    ev=pd.DataFrame(evaluation);ev.to_csv(out/'held_out_prediction_scores.csv',index=False)
    # Neuron-wise phase 3 alignment is descriptive: includes matching quality in table.
    neuron=[]
    mu=profiles[profiles.representation.eq('mu') & profiles.band.ne('uncaptured') & profiles.reference_neuron.ge(0)]
    for (traj,j,band),g in mu.groupby(['trajectory_id','reference_neuron','band']):
        g=g[g.beta_per_dim.gt(0)].sort_values('beta_per_dim');a=g.run_name.map(auc).to_numpy();e=g.energy_abs.to_numpy();dt=np.diff(np.log(g.beta_per_dim))
        for lag in range(4):
            x=np.diff(e)/dt;y=-np.diff(a)/dt
            neuron.append(dict(trajectory_id=traj,reference_neuron=j,band=band,lag=lag,n_pairs=max(0,len(x)-lag),min_alignment=float(g.alignment_abs_r.min()),
                               rho=safe_rho(x[:-lag] if lag else x,y[lag:] if lag else y)))
    pd.DataFrame(neuron).to_parquet(out/'aligned_neuron_lag_correlations.parquet',index=False)
    return d,ev


def plots(profiles,states,importance,out):
    folder=out/'figures';folder.mkdir(exist_ok=True)
    mu=profiles[profiles.representation.eq('mu')]
    # One explicit trajectory, never average unidentified axes across models/seeds.
    trajectory='d8_seed13';p=mu[mu.trajectory_id.eq(trajectory)]
    if p.empty:trajectory=mu.trajectory_id.iloc[0];p=mu[mu.trajectory_id.eq(trajectory)]
    betas=sorted(p.beta_per_dim.unique());chosen=[betas[i] for i in [0,len(betas)//2,len(betas)-2]]
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for ax,beta in zip(axs,chosen):
        g=p[p.beta_per_dim.eq(beta)].pivot(index='neuron_id',columns='band',values='energy_fraction')
        im=ax.imshow(g,aspect='auto',vmin=0,vmax=1);ax.set_xticks(range(len(g.columns)),g.columns,rotation=45);ax.set_title(f'β/d={beta:g}');ax.set_ylabel('model-local neuron')
    fig.colorbar(im,ax=list(axs),label='fraction of centred μ energy',shrink=.7);fig.suptitle(trajectory+' · independently trained models');fig.savefig(folder/'dimension_band_heatmaps.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for band,g in p.groupby('band'):
        grid=g.pivot(index='beta_per_dim',columns='neuron_id',values='energy_abs')
        fig,ax=plt.subplots(figsize=(7,4));im=ax.imshow(np.log10(grid+1e-12),aspect='auto');ax.set_yticks(range(len(grid)),[f'{x:g}' for x in grid.index]);ax.set_xlabel('model-local neuron (not longitudinal identity)');ax.set_ylabel('β/d');ax.set_title(f'{trajectory} {band} log10 absolute energy');fig.colorbar(im,ax=ax);fig.savefig(folder/f'beta_dimensions_{band}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for band,g in p.groupby('band'):axs[0].plot(g.groupby('beta_per_dim').energy_abs.sum().index,g.groupby('beta_per_dim').energy_abs.sum().values,marker='.',label=band)
    g=p.drop_duplicates(['run_name','neuron_id']);
    for j,t in g.groupby('neuron_id'):axs[1].plot(t.beta_per_dim,t.KL_j,label=str(j))
    for ax in axs:ax.set_xscale('symlog',linthresh=5e-6);ax.set_xlabel('β/d');ax.legend(fontsize=7)
    axs[0].set_ylabel('absolute centred μ energy');axs[1].set_ylabel('KL per model-local neuron');fig.tight_layout();fig.savefig(folder/'energy_and_kl.png',dpi=160);plt.close(fig)
    p2=importance[importance.representation.eq('mu') & importance.neuron_id.ge(0)]
    spectral=mu[mu.band.eq('B1')].merge(p2[['run_name','neuron_id','delta_auc_fixed']],on=['run_name','neuron_id'])
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter(spectral.energy_fraction,spectral.delta_auc_fixed,s=5,alpha=.3);ax.set_xlabel('B1 energy fraction, μ');ax.set_ylabel('fixed-probe Δ validation AUC');ax.set_title('Association, not a causal effect');fig.tight_layout();fig.savefig(folder/'B1_vs_task_importance.png',dpi=150);plt.close(fig)


def report(table,meta,profiles,importance,states,ev,out,z_draws):
    lines=['# Spektrale Attribution: Ergebnisse und Grenzen','',
           f'Quelle: `{DEFAULT_SWEEP}`. {len(table)} Modelle, {table.trajectory_id.nunique()} Trajektorien; Dimensionen {sorted(table.latent_dim.unique().tolist())}; vier Seeds. Kein d=10-Sweep gefunden.',
           'Graph: fester Test-Referenzgraph, k=20, normalisierter Laplacian. Symmetrisierung wie graph_spectrum: (W+W.T)/2; einseitige binäre Kanten haben Gewicht 0,5, gegenseitige 1. U und Sample-IDs geprüft.',
           'Bänder: disjunkte vorhandene F0-Log-Bänder, keine Zuordnung anhand gerundeter Grenzwerte. Nahe entartete Unterräume nicht zwischen Bändern geteilt.',pd.DataFrame(meta).to_markdown(index=False),'',
           '## Daten und Definitionen',
           'μ/logvar vorhanden für Train/Valid/Test. Keine gespeicherten z in Posterior-NPZ; z wird neu gezogen und nicht als ursprünglicher AUTOENCODIX-z ausgegeben. Keine gespeicherten Probe-Koeffizienten gefunden; Probes werden mit derselben LogReg(max_iter=1000)-Spezifikation auf Train angepasst. Cancer-AUC ist macro OvO.',
           f'z: {z_draws} feste Ziehungen pro Modell, Mittel der AUCs (keine gemittelten Vorhersagen). KL und Active Units beziehen sich stets auf μ/logvar, nicht auf z-Fourierenergie.',
           'Absolute Energie: Summe quadrierter Koeffizienten nach arithmetischer Sample-Zentrierung. Gesamtenergie enthält den nicht erfassten Rest. Konstantes μ erhält null absolute Energie und undefinierte relative Anteile. B0 enthält alle Nullmoden; beim normalisierten Laplacian kann sample-zentriertes μ dennoch B0-Energie haben.',
           'Neuron-Identitäten sind modelllokal. energy_relative benutzt deskriptives Hungarian-Matching der absoluten Aktivierungskorrelation zum β=0-Modell; Matching-Qualität wird gespeichert. Keine Garantie von neuronaler Kontinuität. Entropie ist Bandentropie, invariant gegen Drehungen innerhalb der Bänder. Spektraler Schwerpunkt beschreibt nur erfasste Energie.',
           'Ablation: Trainmittel einsetzen; fixe Probe und neu trainierte Probe getrennt. Gruppen: obere/untere Hälfte der Train-Varianz und zwei größte Train-Varianzen. Modal-Ridge: innere Alpha-Wahl, äußere fünf Folds mit getrennten Graphknoten; transduktive TEST-Graphanalyse, keine induktive Patientenprognose.',
           '', '## A–C: Spezialisierung, β und Aufgabenrelevanz']
    mu=profiles[profiles.representation.eq('mu') & profiles.band.ne('uncaptured')]
    nonzero=mu.drop_duplicates(['run_name','neuron_id']);nonzero=nonzero[nonzero.var_mu_j>1e-12]
    lines.append(f'Nichtkonstante Neuronen: {len(nonzero)}; mediane Bandentropie {nonzero.spectral_entropy.median():.3f}. Dies beschreibt Konzentration, beweist keine seed-stabile Spezialisierung.')
    joined=mu.merge(importance[importance.representation.eq('mu') & importance.neuron_id.ge(0)][['run_name','neuron_id','delta_auc_fixed','delta_auc_retrained']],on=['run_name','neuron_id'])
    associations=[]
    # Calculate within-model associations; report variability, not pseudoreplicated pooled p-values.
    for band,g in joined.groupby('band'):
        values=[safe_rho(t.energy_fraction,t.delta_auc_fixed) for _,t in g.groupby('run_name')]
        finite=np.asarray(values)[np.isfinite(values)]
        associations.append(dict(band=band,n_models=len(finite),median_rho=float(np.median(finite)) if len(finite) else np.nan))
    lines.append(pd.DataFrame(associations).to_markdown(index=False))
    vars=[]
    singles=importance[importance.representation.eq('mu') & importance.neuron_id.ge(0)].merge(nonzero[['run_name','neuron_id','KL_j','var_mu_j']],on=['run_name','neuron_id'])
    for col in ['KL_j','var_mu_j']:
        a=np.array([safe_rho(g[col],g.delta_auc_fixed) for _,g in singles.groupby('run_name')]);a=a[np.isfinite(a)]
        vars.append(dict(feature=col,n_models=len(a),median_rho=np.median(a) if len(a) else np.nan))
    lines.append(pd.DataFrame(vars).to_markdown(index=False))
    lines+=['','## D–E: Frühwarnung und zusätzlicher Nutzen',
            'Spearman auf Werten und Δ/log-β getrennt, Lags 0–3. Positive AUC loss bedeutet Verschlechterung; für sinkende Energie erwartet man eine negative rho mit AUC loss. Mindestens vier gültige Paare, konstante Reihen undefiniert. Wenige β-Punkte; keine kausale Interpretation.',
            'Vorhersage: zukünftiger AUC-Verlust, ganze Seeds zurückgehalten (alle Dimensionen und β derselben Seed-Gruppe). β, nächster β-Wert und bisherige AUC sind starke Baselines. Die Standardbaseline enthält außerdem Rate, Maximum/Streuung/normierte Entropie der KL_j-Verteilung, Active Units, Effective Rank, LCMC und Trustworthiness. Alle Modelle auf denselben vollständigen Zeilen; Ridge alpha=10 vorab fest. Keine Auswahl anhand Testfold. Nur vier unabhängige Seed-Gruppen innerhalb dieses Datensatzes.',
            ev.to_markdown(index=False)]
    mean=ev.groupby('model').rmse.mean()
    if {'beta_auc_standard','beta_auc_standard_spectral'}.issubset(mean.index):
        delta=mean['beta_auc_standard_spectral']-mean['beta_auc_standard']
        lines.append(f'Mittlere Fold-RMSE-Differenz spektral minus Standardbaseline: {delta:+.5f}; negativ bedeutet Verbesserung. Das ist explorativ; fehlende externe Datensatzvalidierung und mehrere getestete Varianten begrenzen die Aussage.')
    # Paired uncertainty over held seed groups, not beta points.
    scores=ev.pivot(index='held_seed',columns='model',values='rmse')
    diff=(scores.beta_auc_standard_spectral-scores.beta_auc_standard).dropna().to_numpy()
    boot=np.random.default_rng(31).choice(diff,(2000,len(diff)),replace=True).mean(1)
    lines.append(f'Exploratives 95%-Seed-Bootstrap-Intervall für diese RMSE-Differenz: [{np.quantile(boot,.025):+.5f}, {np.quantile(boot,.975):+.5f}] ({len(diff)} Seed-Gruppen; geringe Präzision).')
    alerts=pd.read_parquet(out/'retrospective_alerts.parquet')
    lines.append(alerts.groupby('feature').agg(median_lead=('lead_steps','median'),n_defined=('lead_steps','count'),false_alarm_rate=('false_alarm','mean')).to_markdown())
    lines += ['Frühwarn-Schwellen: 5% anhaltender Merkmalsverlust vs. 2% relativer AUC-Verlust ab erster positiver β-Stufe. Das Anhalten bis zum Ende ist eine retrospektive Definition, kein online verfügbarer Alarm. Fehlalarme zählen nur Alarm ohne späteren AUC-Einbruch. Schwellen sind gewählt, nicht theoretisch garantiert.',
              '', '## F: Controller-Vorbereitung',
              'Kein Controller implementiert oder trainiert. Hohe/niedrige KL und feste-Probe-Relevanz werden im Kandidatentableau modellintern nach Medianen eingeteilt, nicht als bewiesene optimale β-Regel.',
              'Risiken: Informationswanderung, Achsenpermutation, redundante Gruppen, Skalenänderung, unzuverlässige Feedbacksignale und mehr Gesamtrate als Ursache höherer AUC. Frequenzenergie in μ erfasst nicht die ganze Posteriorverteilung. Schutz ganzer Unterräume ist belastbarer als feste Achsennummern.',
              'Die Sweeps sind unabhängig trainierte Modelle, keine zeitliche Neuronenverfolgung; β-Vergleiche bleiben deskriptiv. Für spätere Controller Tests auf unabhängigen Trajektorien und rate-gematchte globale β-Baselines erforderlich.',
              'Offen: externe Daten, echte induktive Graph-Extension, unabhängige Kalibrierung, gespeicherte Original-z/Probe-Objekte, Replikation außerhalb Split 2. Es ist kein Zusammenhang zwischen Einzelneuronen und Graphfrequenzen theoretisch erzwungen.']
    candidates=singles.copy()
    candidates['KL_high']=candidates.KL_j>candidates.groupby('run_name').KL_j.transform('median')
    candidates['task_high']=candidates.delta_auc_fixed>candidates.groupby('run_name').delta_auc_fixed.transform('median')
    candidates['category']=np.select([candidates.KL_high & ~candidates.task_high,~candidates.KL_high & candidates.task_high],['high_KL_low_task','low_KL_high_task'],default='other')
    candidates.to_parquet(out/'controller_candidates.parquet',index=False)
    (out/'spectral_control_findings.md').write_text('\n\n'.join(lines),encoding='utf-8')


def run(sweep=DEFAULT_SWEEP,xfile=DEFAULT_X,out=DEFAULT_OUT,z_draws=3,phase='all'):
    sweep,xfile,out=map(Path,(sweep,xfile,out));table,X,ids,idx,hashes=inventory(sweep,xfile,out)
    clinfile=ROOT.parent/'autoencodix_package/data/raw/OntixTutorialData/combined_clin_formatted.parquet'
    clin=pd.read_parquet(clinfile);clin.index=clin.index.astype(str);labels(clin,ids)
    provenance=dict(sweep=str(sweep),metrics_sha256=digest(sweep/'experiment_metrics.csv'),x_sha256=digest(xfile),clinical_sha256=digest(clinfile),posterior_sha256=hashes,z_draws=z_draws,
                    code_sha256=digest(Path(__file__)),attribution_code_sha256=digest(Path(__file__).parents[1]/'metrics/graph_signals/dimension_attribution.py'))
    manifest=out/'provenance.json'
    if manifest.exists() and json.loads(manifest.read_text())!=provenance:raise ValueError('Provenance changed; use a new output directory rather than stale caches')
    manifest.write_text(json.dumps(provenance,indent=2))
    U,lam,bands,names,meta=basis(X,ids,idx,out);(out/'bands.json').write_text(json.dumps(meta,indent=2))
    print(f'Inventory: {len(table)} runs, X {X.shape}, U {U.shape}, dimensions {sorted(table.latent_dim.unique())}',flush=True)
    if (out/'spectral_dimension_profiles.parquet').exists():
        profiles=pd.read_parquet(out/'spectral_dimension_profiles.parquet');states=pd.read_parquet(out/'spectral_run_summary.parquet')
    else:profiles,states=phase1(table,U,lam,bands,names,ids,sweep,out,z_draws)
    print('Phase 1 complete',flush=True)
    if phase=='profiles':return
    if (out/'dimension_task_importance.parquet').exists():
        importance=pd.read_parquet(out/'dimension_task_importance.parquet');scores=pd.read_parquet(out/'band_reconstructability.parquet')
    else:importance,scores=phase2(table,profiles,U,bands,names,ids,sweep,clin,out,z_draws)
    states,ev=phase3(states,importance,scores,profiles,out)
    plots(profiles,states,importance,out);report(table,meta,profiles,importance,states,ev,out,z_draws)
    print(ev.groupby('model').rmse.mean().to_string(),flush=True);print('Completed:',out,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sweep',type=Path,default=DEFAULT_SWEEP);parser.add_argument('--xfile',type=Path,default=DEFAULT_X);parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--z-draws',type=int,default=3);parser.add_argument('--phase',choices=['all','profiles'],default='all');args=parser.parse_args()
    if args.z_draws<1:parser.error('z-draws must be positive')
    run(args.sweep,args.xfile,args.output,args.z_draws,args.phase)
