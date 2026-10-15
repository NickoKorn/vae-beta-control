# Encoder- und Latentgeometrie ohne Decoder

Die neue Datei `encoder_geometry.py` ergänzt das vorhandene Framework. Es werden
keine Modelle trainiert, keine Decoder-Jacobians und keine Rekonstruktionen erzeugt.
Vorhandene Rekonstruktionsmetriken und AUC dürfen als Vergleichsgrößen einfließen.
Die gesamte Kombination über β und Seeds ist unser experimentelles Design, kein
unverändert übernommenes Verfahren aus einem einzelnen Paper.

## Was ist belegt, was ist eine Analyseentscheidung?

- Encoder-Jacobians als lokale Sensitivitäts-/Kontraktionsmaße: Rifai et al. (2011),
  https://icml.cc/2011/papers/455_icmlpaper.pdf . Das Paper regularisiert den
  Jacobian; wir messen ihn an gespeicherten Modellen, ohne diesen Loss zu trainieren.
- Höhere Encoder-Ableitungen: Rifai et al., Higher Order Contractive Auto-Encoder,
  https://www-labs.iro.umontreal.ca/~vincentp/Publications/CAE_H_ECML2011.pdf .
  Eine Hessian-Norm ist trotzdem keine intrinsische skalare Krümmung.
- Glatte Pullback-Geometrie und zweite Fundamentalform: Geomstats-Dokumentation,
  https://geomstats.github.io/notebooks/09_practical_methods__implement_your_own_riemannian_geometry.html .
  `immersion_curvature` verwendet die euklidische Gauss-Gleichung auf einer
  expliziten C²-Immersion. Keine automatisch geschätzte TCGA-Karte wird unterstellt.
- Fisher-Geometrie diagonaler Normalverteilungen:
  https://geomstats.github.io/_modules/geomstats/information_geometry/normal.html .
- Gaussian W2-Formel: https://doi.org/10.1016/j.jmva.2024.105334 (Gleichung 6).
  Bei diagonalem Σ reduziert sie sich auf die euklidische Metrik in (μ,σ).
- Vierpunkt-Hyperbolizität: Fournier et al., Computing the Gromov hyperbolicity
  of a discrete metric space,
  https://webusers.imj-prg.fr/~herve.fournier/publications/hyperbolicity.pdf .
- Intrinsische skalare Krümmung aus Punktwolken ist grundsätzlich möglich, braucht
  aber einen eigenen konsistenten Schätzer mit Sampling-/Metrikannahmen:
  https://arxiv.org/abs/2308.02615 . Dieser Schätzer wird hier NICHT nachimplementiert.
- VAE-Geodäten in https://arxiv.org/abs/1710.11379 und
  https://proceedings.mlr.press/v151/arvanitidis22b.html sind decoderbasiert und
  keine Belege für unsere reine Encoder-Auswertung.

Die konkreten Stichprobengrenzen, k=5/10/20, 1024 Vierergruppen, Median-Aggregation,
Rangschwelle und β-Vergleiche sind dokumentierte Analyseentscheidungen.

## Mathematische Größen und Interpretationsgrenzen

| Familie | Grundlage / Definition | Checkpoint? | Numerik und Interpretation | Controller-Eignung |
|---|---|---|---|---|
| `encoder_mu_*` | Jμ=∂μ/∂x; σmax, Summe σ², Entropierang exp H(σ/Σσ), Rang, Nullität und Kondition im nichtverschwindenden Spektrum | Ja | Float64, kleines JJᵀ statt 2000×2000-Matrix; harte Grenzen/ReLU liefern nur lokale Ableitungen gemäß Autodiff | Kandidat für Sensitivität, nicht direkt Semantik oder Krümmung |
| `encoder_posterior_w2_*` | Jacobian der Abbildung (μ,σ); G=JμᵀJμ+JσᵀJσ mit Jσ=diag(σ/2)Jlogvar | Ja | Lokale W2-Geometrie einzelner diagonaler q(z\|x), keine Distanz zum Input und kein aggregierter Posterior | Explorativ; trennt Mittelwert- und Varianzänderungen |
| `encoder_posterior_fisher_*` | G=Jμᵀdiag(σ⁻²)Jμ+½JlogvarᵀJlogvar | Ja | Fisher-Sensitivität; unterscheidet sich von W2 und von KL zur Prior | Explorativ; zusätzlicher Nutzen noch offen |
| `encoder_intrinsic_scalar_curvature` | Nur auf verifizierter glatter, nichtdegenerierter Karte definiert | Zusätzlich Karte/Regularität | In der normalen TCGA-Pipeline ausdrücklich `not_defined`, kein Zahlenersatz | Derzeit nicht verwendbar |
| `immersion_curvature` | Zweite Fundamentalform B=H−J(JᵀJ)⁻¹JᵀH; Gauss-Gleichung in metrisch orthonormalem Rahmen | Explizite glatte Abbildung | Volle Rangprüfung, keine Pseudoinverse/Ridge; Sectional-Wert für Rahmenrichtungen 0/1, Scalar volle Spur | Validierungsfunktion; kein automatisch gültiges TCGA-Signal |
| Normen / Varianzspur / Unique- und Zero-Fraction | Unzentrierte Punktnormen; tr Cov mit Varianzdivisor N | Nein | Radialinformation bleibt erhalten; vom Koordinatenursprung/Skalierung abhängig | Kontraktion/Collapse, keine Krümmung |
| Angular-Preservation | arccos des Skalarprodukts normierter Vektoren; Spearman mit ursprünglichen euklidischen Inputdistanzen | Nein | Ganze gepaarte Kohorte bleibt gleich; bei Nullvektoren nicht verfügbar | Formmaß ohne Radialinformation; kein Nachweis einer gelernten Sphäre |
| `delta_lower_bound`, `delta_mean`, `delta_over_diameter` | Halbe Differenz der zwei größten Vierpunkt-Distanzsummen; Maximum/Mittelwert der fixierten Stichprobe | Nein | Maximum ist nur Untergrenze der endlichen Hyperbolizitätskonstante; normiert durch Stichprobendurchmesser; bei völligem Collapse undefinierte Normierung | Kein Beweis hyperbolischer Mannigfaltigkeit oder negativer Sectional Curvature |
| `knn_path_to_chord_median_k*` | Kanonischer Union-kNN-Graph, ursprüngliche euklidische Kantenlängen, Dijkstra/Shortest Path gegenüber direkter Distanz | Nein | Näherung auf begrenzter Stichprobe; keine exakte Riemannsche Geodäte; getrennte Komponenten werden nicht verbunden | Sampling-/Dichtesensitiv; nur zusammen mit Connected-Pair-Fraction/Edge-Density |
| `mu_to_input_distance_ratio_median` | Median dμ/dX auf denselben ungeordneten Probenpaaren mit dX>0 | Nein | Rohe Skalenveränderung, keine künstliche Gleichskalierung; gleiche Vorverarbeitung erforderlich | Kontraktion, nicht automatisch Informationserhalt |
| Graph-Ricci/Forman | Bestehende `graph_curvature_metrics.py` und `graph_metrics_pipeline.py` | Nein | Diskrete Krümmung mit eigenen Gewichten/k/Dichte; bleibt separat | Kandidat, keine Gleichsetzung mit glatter Krümmung |

Singulärwerte werden vollständig gespeichert. Die effektive Rangschwelle ist das
Maximum aus `rank_rtol` und sqrt(Maschinenpräzision), da Gram-Eigenwerte quadrierte
Singulärwerte sind. Der gemeldete `condition_nonzero` ignoriert den Kernel bewusst;
`input_nullity` und `full_input_metric_positive_definite` verhindern eine falsche
Interpretation als nichtdegenerierte Inputmetrik. Erstordnungsmetriken sind bei
ReLU fern der Knicke sinnvoll, aber keine globale C²-Garantie.

## Warum keine gewöhnliche Krümmung aus J_EᵀJ_E?

Für p Inputmerkmale und d<p Encoder-Dimensionen gilt Rang(JμᵀJμ)≤d<p.
Die Matrix ist positiv semidefinit, aber nicht positiv definit. Die gewöhnlichen
Riemannschen Krümmungsformeln verlangen eine reguläre Metrik. Ein künstlicher
Ridge-Term würde eine andere, willkürlich gewählte Geometrie definieren.

Der vorhandene VARIX-Encoder verwendet ReLU, `torch.where` für μ und `clamp` für
logvar. Für μ ist er innerhalb der Aktivierungsbereiche stückweise affin; ein
Hessian von null dort bedeutet nicht, dass die Patientendaten global flach sind.
Auch eine lokale PCA-Ebene wäre ohne zusätzliche Validierung keine bekannte
Parametrisierung der tatsächlichen Datenmannigfaltigkeit.

## Daten, β-Auswertung und Unsicherheit

Sample-IDs werden explizit angeglichen. Der bekannte Inputcache wird unverändert
verwendet; keine neue Skalierung wird anhand von Testlabels geschätzt. Geometrie
verwendet standardmäßig dieselben 128 Patienten für jedes Modell. β, Dimension,
Seed, Rate, Active Units, Effective Rank, bestehende Collapse-Indikatoren,
Rekonstruktionsfehler, Preservation und Cancer-AUC werden per Run verknüpft.

Jacobians werden aus finalen Checkpoints berechnet und gegen gespeicherte μ und
logvar geprüft. Ein fehlender Checkpoint bleibt fehlend; Posterior-Dateien alleine
reichen nicht aus. Der Modellzustand und Trainingsmodus werden nicht verändert.
Stochastisches z wird hier nicht gezogen; vorhandene AUCs aus z werden als solche
gekennzeichnet. Keine Dimensionsidentität über unabhängige Modelle wird angenommen.

Die β-Kurven zeigen je Dimension den Median und IQR über Seeds. Korrelationen
werden innerhalb einzelner Trajektorien gerechnet. Der bestehende segmentierte
Fit liefert explorative Change Points und Fit-Gains, keine inferenzielle oder
kausale Frühwarngarantie. Vor Controller-Einsatz fehlen insbesondere Bestätigung
auf zurückgehaltenen Trajektorien, Fehlalarmmessung und zusätzlicher Vorhersagenutzen
gegenüber β/Rate/bisheriger AUC. Bei einer Seedauswahl von nur vier Seeds sind
Unsicherheiten wesentlich; keine Patienten-Pseudoreplikation.

## Verwendung

Im bestehenden JSON optional `metrics.encoder_geometry` mit `enabled=true` setzen.
Standardmäßig bleibt die zusätzliche Familie deaktiviert. Bestehende Graph- und
Preservation-Metriken werden weiter von ihren bisherigen Modulen verantwortet.

Gespeicherte Daten ohne Training auswerten:

```bash
python -m beta_vae_experiments.result_analysis.encoder_geometry_analysis beta_vae_experiments/user_interface/experiments.json#analysis.encoder_geometry
python -m pytest beta_vae_experiments/test/test_encoder_geometry.py -q
```

Ergebnisse: Parquet-Dateien, ein gemeinsamer β-Plot, Run-/Checkpoint-Inventar,
pro Run Normen, Jacobian-Spektren und Provenienz. Cache-Schlüssel prüfen Input,
Posterior, Checkpoint und Konfiguration. Es werden keine externen Programme
installiert. Geoopt ist höchstens eine spätere Option für ausdrücklich definierte
nicht-euklidische Modelle; eine nachträgliche Projektion belegt keine gelernte
hyperbolische/sphärische Geometrie.
