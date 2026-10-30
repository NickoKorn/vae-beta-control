"""Live dashboard for one preservation-annealing run.

Reads only what the training process writes into its run folder (controller log, epoch
snapshots, controller reference), so it never touches the training itself and also replays
finished runs. Per snapshot it stores scalar diagnostics in ``live_metrics.csv`` and a
plain-language overlap report in ``live_summary.md`` inside the run folder; both are meant
to be read by a researcher or an agent without looking at the plots.

Panels: beta/d, controller signal with target band, rate with epsilon-delta collapse,
cluster overlap (label-free, see ``cluster_overlap``), patient collapse at mu = 0, and the
AUTOENCODIX UMAP (``UMAP`` + ``GeneralVisualizer._plot_2D``) of z for the latest snapshot.
"""

from __future__ import annotations

import io
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from beta_vae_experiments.cluster_overlap import ClusterReference, describe, scalar_summary
from beta_vae_experiments.metrics.information_theoretic import compute_conditional_kl_matrix
from beta_vae_experiments.metrics.latent_usage import PosteriorCollapseConfig

PRESENTATION = ("BRCA", "COAD", "LUAD", "LUSC", "OV", "READ", "UCEC", "UCS")
COLLAPSE = PosteriorCollapseConfig()


def find_run_folder(path):
    """Accept either the run folder itself or the experiment folder above ``run_analysis``."""
    path = Path(path)
    if (path / "controller_log.csv").exists() or (path / "epoch_snapshots").exists():
        return path
    folders = sorted((path / "run_analysis").glob("*/"))
    return folders[-1] if folders else None


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None  # Not written yet, or the training process is writing it right now.


def _figure_image(figure, dpi=100):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return plt.imread(buffer)


class LiveDashboard:
    def __init__(self, path, label="CANCER_TYPE_ACRONYM", focus=PRESENTATION, umap_splits=("train", "valid"), k=20):
        self.path, self.label, self.focus, self.umap_splits, self.k = Path(path), label, focus, umap_splits, k
        self.run = None
        self.reference = self.ids = self.cluster_names = None
        self.metrics = pd.DataFrame()
        self.descriptions = {}
        self.umaps = {}
        self._figure_handle = self._text_handle = None
        self._last_draw = 0.0

    # ----------------------------------------------------------------- data
    def _attach(self):
        if self.run is None:
            self.run = find_run_folder(self.path)
        if self.run is None or self.reference is not None:
            return
        reference_path = self.run / "controller_validation_reference.npz"
        if not reference_path.exists():
            return
        with np.load(reference_path) as saved:
            x, self.ids = saved["x"].astype(np.float64), saved["sample_ids"].astype(str)
        cache = self.run / "cluster_reference.npz"
        if cache.exists():
            with np.load(cache) as saved:
                self.reference = ClusterReference(x, k=self.k, labels=saved["labels"],
                                                  silhouettes=pd.Series(saved["silhouettes"], index=saved["candidates"]))
        else:
            self.reference = ClusterReference(x, k=self.k)
            np.savez(cache, labels=self.reference.labels, silhouettes=self.reference.silhouettes.to_numpy(),
                     candidates=self.reference.silhouettes.index.to_numpy())
        previous = _read_csv(self.run / "live_metrics.csv")
        if previous is not None:
            self.metrics = previous
        if (self.run / "live_descriptions.json").exists():
            self.descriptions = {int(e): t for e, t in json.loads((self.run / "live_descriptions.json").read_text()).items()}

    def _names(self):
        labels_path = self.run / "epoch_snapshots" / "labels_valid.parquet"
        if self.cluster_names is not None or not labels_path.exists():
            return self.cluster_names
        labels = pd.read_parquet(labels_path)
        labels.index = labels.index.astype(str)
        if self.label not in labels:
            return None
        values = labels.reindex(self.ids)[self.label].fillna("unbekannt")
        self.cluster_names = {c: ", ".join(f"{name} {share:.0%}" for name, share in
                                           values[self.reference.labels == c].value_counts(normalize=True).head(2).items())
                              for c in range(self.reference.n_clusters)}
        return self.cluster_names

    def snapshot_epochs(self):
        if self.run is None:
            return []
        folder = self.run / "epoch_snapshots"
        return sorted(int(p.stem.split("_")[1]) for p in folder.glob("epoch_*_valid.npz")
                      if (folder / p.name.replace("_valid", "_train")).exists())

    def _process(self, epoch):
        with np.load(self.run / "epoch_snapshots" / f"epoch_{epoch:05d}_valid.npz") as saved:
            mu, logvar, z = saved["mu"], saved["logvar"], saved["z"]
            positions = pd.Index(saved["sample_ids"].astype(str)).get_indexer(self.ids)
        if (positions < 0).any():
            raise ValueError("Snapshot enthält nicht alle Patienten der Controller-Referenz.")
        kl = compute_conditional_kl_matrix(mu, logvar)
        subset = mu[positions].astype(np.float64)
        result = self.reference.evaluate(subset)
        row = dict(epoch=epoch, rate=float(kl.sum(1).mean()),
                   collapsed_dimensions=int(np.sum(np.mean(kl < COLLAPSE.primary_epsilon, axis=0) >= 1 - COLLAPSE.delta)),
                   distinct_mu_share=len(np.unique(subset, axis=0)) / len(subset),
                   max_zero_share=float((mu == 0).mean(axis=0).max()),
                   same_cluster_neighbours_z=self.reference.same_cluster_share(z[positions].astype(np.float64)),
                   **scalar_summary(result))
        return row, describe(result, self._names())

    def update_metrics(self):
        """Process all snapshots not yet in ``live_metrics.csv``. Returns the newly processed epochs."""
        self._attach()
        if self.reference is None:
            return []
        done = set(self.metrics.epoch) if len(self.metrics) else set()
        new = []
        for epoch in self.snapshot_epochs():
            if epoch in done:
                continue
            try:
                row, text = self._process(epoch)
            except (OSError, ValueError, KeyError, EOFError):
                break  # Snapshot still being written; try again on the next refresh.
            self.metrics = pd.concat([self.metrics, pd.DataFrame([row])], ignore_index=True).sort_values("epoch")
            self.descriptions[epoch] = text
            new.append(epoch)
        if new:
            self.metrics.to_csv(self.run / "live_metrics.csv", index=False)
            (self.run / "live_descriptions.json").write_text(json.dumps({str(e): t for e, t in self.descriptions.items()},
                                                                       ensure_ascii=False, indent=1))
            (self.run / "live_summary.md").write_text(self.summary())
        return new

    def controller(self):
        return _read_csv(self.run / "controller_log.csv") if self.run is not None else None

    def umap_image(self, epoch):
        """AUTOENCODIX-style UMAP of z (train + validation) for one snapshot, coloured by ``label``."""
        if epoch in self.umaps:
            return self.umaps[epoch]
        from umap import UMAP
        from autoencodix.visualize._general_visualizer import GeneralVisualizer
        frames, labels = [], []
        for split in self.umap_splits:
            with np.load(self.run / "epoch_snapshots" / f"epoch_{epoch:05d}_{split}.npz") as saved:
                frames.append(pd.DataFrame(saved["z"], index=saved["sample_ids"].astype(str)))
            annotation = pd.read_parquet(self.run / "epoch_snapshots" / f"labels_{split}.parquet")
            annotation.index = annotation.index.astype(str)
            labels.append(annotation.reindex(frames[-1].index)[self.label].fillna("unknown"))
        z, labels = pd.concat(frames), pd.concat(labels)
        embedding = pd.DataFrame(UMAP(n_components=2, random_state=0).fit_transform(z))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            figure = GeneralVisualizer._plot_2D(embedding=embedding, labels=labels.tolist(), focus_labels=list(self.focus),
                                                param=self.label, layer=f"UMAP von z, Epoche {epoch}", figsize=(12, 8), center=True)
        self.umaps[epoch] = _figure_image(figure)
        return self.umaps[epoch]

    # ----------------------------------------------------------------- text
    def summary(self, epoch=None):
        """Plain-language status of the latest (or a chosen) snapshot, also stored as ``live_summary.md``."""
        if not len(self.metrics):
            return "Noch kein Snapshot ausgewertet."
        row = self.metrics.set_index("epoch").loc[epoch if epoch is not None else self.metrics.epoch.max()]
        epoch = int(row.name)
        lines = [f"**Snapshot Epoche {epoch}**"]
        controller = self.controller()
        if controller is not None:
            controller = controller[controller.epoch <= epoch]
        if controller is not None and len(controller):
            latest = controller.iloc[-1]
            measured = controller.dropna(subset=["metric_value"])
            signal = f"{measured.iloc[-1].metric_value:.3f} (Epoche {int(measured.iloc[-1].epoch)})" if len(measured) else "Warm-up"
            lines.append(f"Controller bei Epoche {int(latest.epoch)}: Zustand `{latest.state}`, β/d = {latest.beta_over_d:.2e}, "
                         f"{latest.metric} = {signal}, Ziel {latest.target} ± {latest.tolerance}.")
        lines.append(f"Rate {row.rate:.2f} Nats, {int(row.collapsed_dimensions)} kollabierte Dimensionen (ε-δ), "
                     f"{100 * row.distinct_mu_share:.0f} % distinkte Patienten in μ, bis zu {100 * row.max_zero_share:.0f} % μ = 0 "
                     f"in einer Dimension.")
        lines.append(f"Eigener X-Cluster unter den {self.k} Nachbarn in z: {100 * row.same_cluster_neighbours_z:.0f} %.")
        return "\n\n".join(lines + [self.descriptions.get(epoch, "").replace("\n", "\n\n")])

    # ----------------------------------------------------------------- plots
    def figure(self, umap_epoch=None):
        controller = self.controller()
        metrics = self.metrics
        fig = plt.figure(figsize=(17, 13))
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.9])
        ax = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]
        if controller is not None and len(controller):
            ax[0].step(controller.epoch, controller.beta_over_d.clip(lower=1e-8), where="post", color="black")
            ax[0].set(yscale="log", title="β/d", xlabel="Epoche")
            measured = controller.dropna(subset=["metric_value"])
            ax[1].plot(measured.epoch, measured.metric_value, "o-", ms=2.5)
            last = controller.iloc[-1]
            if pd.notna(last.target):
                ax[1].axhspan(last.target - last.tolerance, last.target + last.tolerance, alpha=.15)
            ax[1].set(title=f"Steuersignal: {last.metric}", xlabel="Epoche")
        if len(metrics):
            ax[2].plot(metrics.epoch, metrics.rate, "o-", ms=3)
            collapsed = metrics[metrics.collapsed_dimensions > 0]
            ax[2].plot(collapsed.epoch, collapsed.rate, "x", color="black", ms=8, mew=2, label="≥ 1 kollabierte Dim.")
            ax[2].set(yscale="log", title="Rate [Nats] (× = ε-δ-Collapse)", xlabel="Epoche")
            if len(collapsed):
                ax[2].legend(fontsize=8)
            ax[3].plot(metrics.epoch, metrics.same_cluster_neighbours_latent, "o-", ms=3, label="μ")
            ax[3].plot(metrics.epoch, metrics.same_cluster_neighbours_z, "s--", ms=3, label="z")
            ax[3].axhline(metrics.same_cluster_neighbours_x.iloc[0], color="gray", ls=":", label="X")
            ax[3].axhline(metrics.chance.iloc[0], color="red", ls=":", lw=.8, label="Zufall")
            ax[3].set(ylim=(0, 1), title=f"Nachbarn im eigenen X-Cluster (k = {self.k})", xlabel="Epoche")
            ax[3].legend(fontsize=8)
            ax[4].plot(metrics.epoch, metrics.recovery_ari, "o-", ms=3, label="ARI: X-Cluster wiedergefunden")
            ax[4].plot(metrics.epoch, metrics.silhouette_x_clusters_in_latent, "s--", ms=3, label="Silhouette der X-Cluster")
            ax[4].axhline(0, color="gray", lw=.8)
            ax[4].set(ylim=(-.5, 1), title=f"Cluster-Erhalt (X: {int(metrics.n_clusters_x.iloc[0])} Cluster)", xlabel="Epoche")
            ax[4].legend(fontsize=8)
            ax[5].plot(metrics.epoch, metrics.distinct_mu_share, "o-", ms=3, label="distinkte Patienten in μ")
            ax[5].plot(metrics.epoch, metrics.max_zero_share, "s--", ms=3, label="max. Anteil μ = 0 je Dimension")
            ax[5].set(ylim=(0, 1.02), title="Patienten-Collapse (μ-Klemme)", xlabel="Epoche")
            ax[5].legend(fontsize=8)
            umap_epoch = umap_epoch if umap_epoch is not None else int(metrics.epoch.max())
            for axis in ax:
                axis.axvline(umap_epoch, color="purple", lw=.8, alpha=.5)
            image_axis = fig.add_subplot(grid[2, :])
            image_axis.imshow(self.umap_image(umap_epoch))
            image_axis.set_axis_off()
        fig.suptitle(f"Live-Dashboard: {self.path.name}", fontsize=12)
        fig.tight_layout()
        return fig

    def show(self):
        """Create the display slots; call before ``refresh``."""
        self._figure_handle = display(Markdown("_Warte auf den ersten Snapshot …_"), display_id=True)
        self._text_handle = display(Markdown(""), display_id=True)
        return self

    def refresh(self, min_interval=15.0, force=False):
        """Update metrics and redraw if a new snapshot arrived or ``min_interval`` seconds passed."""
        new = self.update_metrics()
        if self._figure_handle is None or not len(self.metrics):
            return new
        if new or force or time.time() - self._last_draw >= min_interval:
            figure = self.figure()
            self._figure_handle.update(figure)
            plt.close(figure)
            self._text_handle.update(Markdown(self.summary()))
            self._last_draw = time.time()
        return new

    def browse(self):
        """Replay a finished run: slider over the snapshots, UMAP and overlap report for the chosen epoch."""
        import ipywidgets as widgets
        self.update_metrics()
        epochs = [int(e) for e in self.metrics.epoch]
        output = widgets.Output()

        def render(epoch):
            with output:
                output.clear_output(wait=True)
                figure = self.figure(umap_epoch=epoch)
                display(figure)
                plt.close(figure)
                display(Markdown(self.summary(epoch)))

        slider = widgets.SelectionSlider(options=epochs, value=epochs[-1], description="Epoche", continuous_update=False,
                                         layout=widgets.Layout(width="90%"))
        slider.observe(lambda change: render(change["new"]), names="value")
        display(slider, output)
        render(epochs[-1])
