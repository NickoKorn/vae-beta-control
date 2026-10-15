# Zentrale Versuchskonfiguration

**experiments.json** enthält acht aktuelle Profile. Die 15
historischen beziehungsweise technischen Varianten stehen in **presets_archive.json**.
Alle 23 Profilnamen bleiben über die zentrale Datei aufrufbar; das Archiv wird
nur beim Abruf eines dort liegenden Profils geladen.
Gemeinsame Werte gelten innerhalb einer Gruppe als `defaults`. Ein Profil enthält
nur seine Abweichungen. Dictionaries werden rekursiv zusammengeführt; Listen
werden vollständig ersetzt. Ein Profil verändert kein anderes Profil.

| Gruppe | Inhalt |
|---|---|
| `training` | Aktuell: TCGA, Circle, Sphere, Swiss Roll, TCGA ohne Annealing und β/d-Vergleich; ältere Zell-/Konvergenz-Sweeps im Archiv |
| `smoke` | Kurze technische Tests und Export-/Graph-Prüfungen |
| `annealing` | Fester Annealing-/Recovery-Versuchsplan |
| `analysis` | Auswertung gespeicherter Encoder-/Latentgeometrie; kein VAE-Training |
| `step_response` | Historische TCGA-Schrittantwort-Konfigurationen; kein neuer Metrik-Controller |

Die Gruppierung verändert keine Seeds, β-Werte, Epochenzahlen, Datensplits oder
Ergebnisverzeichnisse. Historische Profile bleiben erhalten. Insbesondere enthält
die Datei noch keinen validierten neuen β-Feedback-Controller.

## Anzeigen und prüfen – startet kein Training

Aus dem Repository mit aktivierter Projektumgebung:

```bash
python -m beta_vae_experiments.user_interface.config --list
python -m beta_vae_experiments.user_interface.config 'beta_vae_experiments/user_interface/experiments.json#training.tcga'
python -m beta_vae_experiments.new_beta_varix_experiments 'beta_vae_experiments/user_interface/experiments.json#training.circle' --preflight
```

Ohne `#gruppe.profil` wird das bisherige Standardexperiment `training.tcga`
gewählt. Die reine `config`-CLI druckt nur Einstellungen. Der eigentliche
Trainingsrunner kann umfangreiche Läufe starten, wenn `--preflight` fehlt.

Ein expandiertes Trainingsprofil kann exportiert werden. Seine relativen
Ergebnisverzeichnisse werden dafür auf die ursprünglichen absoluten Pfade
aufgelöst:

```bash
python -m beta_vae_experiments.user_interface.config 'beta_vae_experiments/user_interface/experiments.json#training.circle' --output /tmp/circle.json
```

Eigene vollständige JSON-Dateien funktionieren weiterhin mit dem bisherigen Runner.
Die drei Standardprofile im Bash-Starter sind auf die zentrale Datei umgestellt.

## Annealing vorbereiten – startet kein Training

```bash
python -m beta_vae_experiments.result_analysis.prepare_annealing_recovery 'beta_vae_experiments/user_interface/experiments.json#annealing.recovery' --output-dir /tmp/annealing-design
```

Der Plan erzeugt eigenständig ausführbare JSONs und β-Vorschauen. Erst der
Trainingsrunner beziehungsweise der gesonderte Bash-Starter führt diese aus.

## Gespeicherte Geometrie auswerten

```bash
python -m beta_vae_experiments.result_analysis.encoder_geometry_analysis 'beta_vae_experiments/user_interface/experiments.json#analysis.encoder_geometry'
```

Der Neighborhood-Intervention-Pilot wurde lokal archiviert. Für neue Experimente
verwenden wir den zentralen β-Annealer in `beta_annealing.py`.

## Datenartefakte

Die frühere `split.json` enthält Patienten-IDs und ist kein Versuchsparameter.
Sie liegt unverändert unter `datasets/manifests/tcga_split.json`.
Das Zenodo-Downloadmanifest liegt unter
`datasets/manifests/tcga_pancancer_source.json`. Die `resources`-Einträge im Katalog
verweisen auf beide Dateien. Download und Patienten-Splits werden nicht automatisch
beim Lesen einer Konfiguration ausgeführt oder verändert.

`read_config()` liefert neben den Einstellungen deren ursprüngliches
Bezugsverzeichnis. Dadurch behalten insbesondere die früheren Profile aus
`smoke_downstream_graph/` ihre bisherigen relativen Ergebnisverzeichnisse.
