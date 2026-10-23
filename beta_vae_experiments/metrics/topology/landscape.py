"""GUDHI landscapes of the existing diameter-normalized finite VR diagrams.

Literatur: Bubenik (2015), Statistical Topological Data Analysis using
Persistence Landscapes, https://jmlr.org/papers/v16/bubenik15a.html
Die Arbeit begründet Funktionsrepräsentationen für statistische Auswertung,
nicht gerade unsere Anzahl/Rasterauflösung oder einen AUC-Vorhersagewert.
Implementierung: https://gudhi.inria.fr/python/3.11.0/representations.html
Landscapes bleiben explorative Artefakte; keine zusätzlichen Core-Prädiktoren.
"""
from ._diagrams import _finite_diagram

from dataclasses import dataclass, asdict
from importlib.metadata import version
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from gudhi.representations import Landscape

LANDSCAPE_NAMES = tuple(f'topology_landscape_h{k}_l2' for k in (0, 1))
NORMALIZATION = 'Ripser VR: D_X/max(D_X), D_mu/max(D_mu), separately; no second normalization'

@dataclass(frozen=True)
class LandscapeConfig:
    """Vorläufige Diagnoseeinstellungen, keine nachgewiesenen Optimalwerte.

    Fünf Funktionen begrenzen die Auswertung auf die ersten fünf Landscape-
    Ebenen; weitere Ebenen bleiben unberücksichtigt. 100 Punkte diskretisieren
    deren Definitionsbereich. Bei statistischer Nutzung Anzahl/Auflösung auf
    Entwicklung prüfen (z. B. 100 gegen 200) und ein runübergreifend festes
    Raster verwenden. Für reine Einzelrun-Diagnostik ist das paarweise gemeinsame
    Raster zulässig. Rohdiagramme erlauben spätere Offline-Neuberechnung.
    """

    num_landscapes: int = 5
    resolution: int = 100

    def __post_init__(self):
        for key, minimum in [('num_landscapes', 1), ('resolution', 2)]:
            value = getattr(self, key)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')


def compare_landscapes(pd_x, pd_mu, config=LandscapeConfig()):
    """Input representation: the very same normalized VR diagrams as Wasserstein.

    Reuse the core finite-row policy; never cap essential bars or refit per space.

    A joint fit prevents different X/μ grids within one pair. The range is,
    however, re-determined on every call: grids of different runs are not
    automatically comparable. The computed Euclidean vector norm is NOT a
    continuous landscape L2 integral (the grid spacing is missing); GUDHI's
    sqrt(2) scaling is retained. Changing the resolution changes this norm.
    Empty functional representations are mathematically zero functions; if
    the library's data-dependent fit of two empty diagrams yields no finite
    range, we flag the current output as undefined. This is an implementation
    convention, not undefined PH itself.

    Parameters:
        pd_x, pd_mu: Persistence diagram of one homology dimension for X and μ,
            each ONE array of shape (n, 2). Every row is a bar [birth, death];
            the rows are not separate arrays but views on the same memory.
            Essential bars are stored as death=inf.
            Ripser already returns empty diagrams as (0, 2).
            reshape(-1, 2) re-reads the values row-wise in pairs of two. It
            never truncates (incompatible element count -> ValueError), but it
            silently rearranges any even element count: a (2, 3) diagram
            [birth, death, dim] (giotto format) would become wrong (3, 2) bars.
            It only serves to bring an empty list [] with shape (0,) to (0, 2).
            For our Ripser diagrams it has no effect.
        config: Number of landscape levels and grid resolution, see LandscapeConfig.

    Returns:
        (value, vectors, metadata): Euclidean norm of the X-μ vector difference
        (NaN for degenerate input), the two GUDHI vectors of shape
        (2, num_landscapes*resolution), and provenance metadata.
    """
    started = time.perf_counter()
    originals = [np.asarray(p, dtype=float).reshape(-1, 2) for p in (pd_x, pd_mu)]
    finite = [_finite_diagram(p) for p in originals]
    metadata = dict(package='gudhi', version=version('gudhi'),
        api='gudhi.representations.Landscape(**config).fit_transform([pd_x_finite, pd_mu_finite])',
        config=asdict(config), normalization=NORMALIZATION, shared_fit=True,
        infinity_policy='existing topology._finite_diagram: remove rows with any nonfinite coordinate',
        removed_nonfinite=[len(a)-len(b) for a,b in zip(originals,finite)],
        removed_infinite=[int(np.isinf(p).any(axis=1).sum()) for p in originals],
        removed_nan=[int(np.isnan(p).any(axis=1).sum()) for p in originals],
        finite_points=[len(p) for p in finite],
        distance='Euclidean norm of GUDHI sampled vectors, not a continuous integral; library sqrt(2) scaling retained',
        status='ok', reason=None)
    if any((p[:,1] < p[:,0]).any() for p in finite):
        raise ValueError('Persistence death precedes birth')
    transformer = Landscape(**asdict(config))
    vectors = transformer.fit_transform(finite)
    grid = np.asarray(transformer.grid_)
    metadata['grid'] = [float(x) if np.isfinite(x) else None for x in grid]
    metadata['sample_range'] = [float(x) if np.isfinite(x) else None for x in transformer.sample_range_fixed_]
    if not np.isfinite(grid).all() or not np.isfinite(vectors).all():
        value = float('nan')
        metadata.update(status='degenerate_input', reason='Joint GUDHI fit has no finite usable domain (both finite diagrams empty) or nonfinite vectors')
    else:
        value = float(np.linalg.norm(vectors[0]-vectors[1]))
        if np.ptp(grid)==0:
            metadata['reason'] = 'Zero-width finite shared grid; library zero-lifetime landscape retained'
    metadata['runtime_seconds'] = time.perf_counter()-started
    return value, vectors, metadata


def run_landscape_metrics(diagrams, n_samples, output_dir=None, provenance=None, config=LandscapeConfig()):
    """Input representation: cached normalized H0/H1 X/mu PDs; no PH computation."""
    rows=[]
    directory=None if output_dir is None else Path(output_dir)
    if directory: directory.mkdir(parents=True, exist_ok=True)
    metadata=dict(provenance or {}, normalization=NORMALIZATION, homology={})
    for k,name in enumerate(LANDSCAPE_NAMES):
        if diagrams is None:
            value=float('nan');vectors=None
            info=dict(status='missing_diagrams',reason='No normalized VR diagrams available',runtime_seconds=0.)
        else:
            value,vectors,info=compare_landscapes(diagrams[f'topology_input_h{k}_persistence_diagram'],
                diagrams[f'topology_latent_h{k}_persistence_diagram'],config)
        rows.append(dict(metric_name=name,value=value,status=info['status'],warning=info['reason'],n_samples=n_samples,split='test'))
        metadata['homology'][str(k)]=info
        if directory and vectors is not None:np.save(directory/f'landscape_h{k}_vectors.npy',vectors)
    frame=pd.DataFrame(rows)
    if directory:
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        frame.to_csv(directory/'scalar_metrics.csv',index=False)
    return frame
