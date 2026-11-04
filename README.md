# VAE beta control

Masterarbeitsprojekt zur Frage, wie β-Kompression die Struktur und nutzbare
Information im latenten Raum eines Variational Autoencoders verändert.
Die Experimente verwenden VARIX aus [AUTOENCODIX](https://github.com/jan-forest/autoencodix_package)
mit fest gepinnter Paketversion; der AUTOENCODIX-Code bleibt unverändert.

## Datensätze

- **TCGA Pan-Cancer:** Genexpression und DNA-Methylierung mit klinischen
  Annotationen. Downstream-Aufgaben umfassen unter anderem Krebsart,
  Geschlecht, Tumorstadium und Alter, jeweils auf der passenden Patientenkohorte.
- **Circle, Sphere und Swiss Roll:** synthetische Geometrien, um den Erhalt
  bekannter Strukturen unter Kompression zu untersuchen. Die jüngste große
  Fixed-β-Serie verwendet Circle und Swiss Roll neben TCGA.

Zusätzlich stellt `beta_vae_experiments/datasets/gudhi_data.py` die JSON-Dataset-IDs
`gudhi_torus`, `gudhi_spiral`, `gudhi_bunny` und `gudhi_activities` bereit.
GUDHI-Arrays werden zu Pandas-Tabellen mit Sample-IDs und über `DataPackage`
an AUTOENCODIX übergeben. Aktivitätslabels bleiben separate Annotationen.
Remote-Daten werden lokal gecacht; Bunny und Aktivitäten benötigen die Option
`accept_license: true` nach Prüfung der jeweiligen Lizenz.

```python
from beta_vae_experiments.datasets import load_gudhi_dataset

dataset = load_gudhi_dataset("gudhi_torus", {"n_samples": 5000, "dim": 2})
frame = dataset.x_raw                 # Pandas: Patienten/Punkte × Features
data = dataset.to_data_package()      # Eingabe für AUTOENCODIX/VARIX
```

## Experimente

Fixed-β-Sweeps bilden den Zusammenhang zwischen Kompressionsdruck, Rate,
Rekonstruktion, geometrischer Preservation und Downstream-Leistung ab.
Annealing-Experimente vergleichen vorgegebene β-Verläufe mit einer Anpassung
an Preservation-Signale, insbesondere Trustworthiness und Cohesiveness.
Downstream-Evaluationen auf gesampeltem z prüfen den Informationserhalt;
Rate und ε-δ-Diagnosen ordnen Posterior Collapse ein.

Gespeicherte Posteriors und Epoch-Snapshots ermöglichen nachträgliche
Auswertungen von Preservation, kNN-Graphen, Topologie und Posterior-Vermischung.
Notebooks dienen der Analyse und dem Vergleich der Verläufe.

## Wesentliche Fragen

- Wie stark lässt sich der latente Raum komprimieren, bevor Downstream-Leistung
  verloren geht, und unterscheiden sich die Aufgaben dabei?
- Liefern Preservation-Maße zusätzliche Information über die AUC gegenüber β
  und Rate allein?
- Was erklären lokale und globale Strukturveränderungen sowie zunehmende
  Posterior-Vermischung über den Informationsverlust?
- Kann ein labelfreies Preservation-Signal β so begrenzen, dass nützliche
  Information erhalten bleibt und Posterior Collapse vermieden wird?
- Verändert Annealing das Ergebnis gegenüber Training mit demselben festen End-β?

## Experiment-Starter

Mit eingerichteter Projektumgebung lassen sich JSON-Konfigurationen zuerst
prüfen und anschließend als Suite ausführen:

```bash
bash beta_vae_experiments/bash_scripts/run_experiments.sh --preflight /pfad/zur/config_suite
bash beta_vae_experiments/bash_scripts/run_experiments.sh --run /pfad/zur/config_suite
```

Der Starter führt die Konfigurationen nacheinander aus und benötigt für das
Training CUDA. Datensätze, β-Werte, Seeds und Metriken werden in den jeweiligen
JSON-Konfigurationen festgelegt.
