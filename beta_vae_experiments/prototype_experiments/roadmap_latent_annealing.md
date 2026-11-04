# Roadmap: Latentraum-Analysen mit Annealing

Stand: Teil A 31.10.2026, Teil B 03.11.2026. Kennzeichnung jeder Aussage: **gemessen** (mit Umfang), **hergeleitet** (Rechnung oder Code),
**Quelle**, **Vermutung** (nicht geprüft).

Setup aller neuen Läufe: AUTOENCODIX Varix (gepinnt, unverändert), TCGA (`TCGAPanCancerAutoencodix`, MINMAX,
k_filter 2.000 Features, Split-Seed 2), d = 8, Lernrate 5·10⁻⁴, Warm-up 100 Epochen bei β = 0, Controller-Subset
800 Validierungspatienten. Downstream nur auf z. Keine MI als Kontroll- oder Testgröße.

---

## Teil A: Was lässt sich analytisch bestimmen?

Drei Stufen: (A1) exakte Identitäten, sobald μ und σ eines Snapshots vorliegen; (A2) Vorhersagen nur aus den Daten,
ohne Training; (A3) Vorhersagen aus einem Snapshot für Größen, die sonst Sampling oder Klassifikatoren brauchen.

### A1. Exakt aus μ, σ (keine Annahmen außer den Varix-Klemmen)

| Größe | Formel | Status |
|---|---|---|
| Rate pro Dimension | R_d = ½ log(1 + SNR_d) + ½(m_d² + V_d − 1 − log V_d), V_d = Var(μ_d) + σ², SNR_d = Var(μ_d)/E[σ_d²] | hergeleitet (Hoffman & Johnson 2016, Gl. 13–16; Cover & Thomas 2006, Thm. 8.6.5); in jedem Snapshot numerisch geprüft (assert in `trustworthiness_annealer.ipynb`, 6.2) |
| Informationsschranke | I(n; z_d) ≤ ½ log(1 + SNR_d); gemeinsam ½ log det(I + Cov(μ)/σ²) | hergeleitet, wie oben |
| Rate bei fester Klemme | logvar = 0,01 ⇒ R = ½ E‖μ‖² + d · ½(e^0,01 − 1 − 0,01) = ½ E‖μ‖² + d · 2,5·10⁻⁵ | hergeleitet; logvar weicht ab Epoche 300 um ≤ 2·10⁻¹⁰ von 0,01 ab (gemessen, 3 Läufe `trustworthiness_09`) |
| Rate bei vollständigem Collapse | μ ≡ 0 ⇒ R = d · 2,5·10⁻⁵ Nats (Untergrenze, nie 0) | hergeleitet; gemessen im β-Sweep: 5·10⁻⁵·d bei allen 24 Trajektorien |
| Anteil Information an der Rate | Schranke / Rate | gemessen ab Epoche 300: ≈ 65 % bei Ziel 0,9 und 0,7 (`trustworthiness_09`) |

**Folge (hergeleitet):** Die Rate ist in Varix eine reine Funktion der μ-Momente. Sie zu *messen* ist trivial; die
eigentliche Frage ist, sie *vorherzusagen*.

### A2. Vorhersage aus den Daten allein: lineare Gauß-Theorie

**Herleitung der effektiven Rauschvarianz.** AUTOENCODIX mittelt beide Terme (`_base_loss.py:53` `MSELoss(reduction="mean")`,
`:138` KL mit `torch.mean`): L = (1/D) Σ_j (x_j − x̂_j)² + β · (1/d) Σ_k KL_k. Multipliziert mit d/(βD) ist das die
negative Log-Likelihood eines Gauß-Decoders mit Varianz

  s = (β/d) · D / 2   (D = 2.000 Features).

**Quelle:** Der lineare VAE hat dieselben globalen Optima wie probabilistische PCA (Lucas et al. 2019, *Don't Blame the
ELBO*; Tipping & Bishop 1999). Eine Richtung mit Daten-Eigenwert λ_i bleibt nur aktiv, wenn λ_i > s.

**Vorhersage 1: Collapse-Schwellen.** Dimension i kollabiert ab β/d* = 2 λ_i / D. Mit d = 8 sind nur λ_1 … λ_8 relevant.

**Vorhersage 2: Information gegen β.** Für eine Gauß-Quelle ist die Rate-Distortion-Funktion durch Reverse Water-Filling
gegeben (Cover & Thomas 2006, Thm. 10.3.3): I(β) = Σ_{λ_i > s} ½ log(λ_i / s). Vergleichsgröße ist die gemessene
Informationsschranke aus A1, nicht die Gesamtrate.

**Erster Test (gemessen; λ aus den 800 Validierungspatienten des Controller-Subsets, nicht aus Train):**

| i | λ_i | vorhergesagt β/d* = 2λ_i/D |
|---|---|---|
| 1 | 10,70 | 1,1·10⁻² |
| 2 | 6,86 | 6,9·10⁻³ |
| 3 | 3,65 | 3,7·10⁻³ |
| 4 | 3,11 | 3,1·10⁻³ |
| 5 | 2,65 | 2,7·10⁻³ |
| 6 | 2,09 | 2,1·10⁻³ |
| 7 | 1,87 | 1,9·10⁻³ |
| 8 | 1,21 | 1,2·10⁻³ |

Gemessen in den Annealer- und Fest-β-Läufen (Block A und C, 12 Läufe mit Collapse; höchstes β/d bis zur k-ten
kollabierten Dimension, Median [Spanne]):

| k-te Dimension | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| β/d gemessen | 3,7·10⁻³ [3,0–4,6] | 3,7·10⁻³ [3,0–5,8] | 3,8·10⁻³ | 3,9·10⁻³ | 4,6·10⁻³ | 5,8·10⁻³ |
| β/d linear | 1,2·10⁻³ | 1,9·10⁻³ | 2,1·10⁻³ | 2,7·10⁻³ | 3,1·10⁻³ | 3,7·10⁻³ |

Dazu gemessen: festes β/d = 1,5·10⁻³ kollabiert in keinem Seed (linear vorhergesagt: 1 Dimension); alle 8 Dimensionen
erst im β-Sweep zwischen 4·10⁻³ und 1,5·10⁻² (linear: 1,1·10⁻²; anderer Datensatz, 90 Epochen).

**Einordnung.** *Gemessen:* Die lineare Theorie trifft die Größenordnung (10⁻³ … 10⁻²), aber die ersten Dimensionen
kollabieren später als vorhergesagt (3,7 statt 1,2·10⁻³) und gebündelt statt gestaffelt. *Vermutung:* Der nichtlineare
Decoder nutzt schwache Richtungen effizienter als PCA; die Klemmen (σ² ≥ 1, μ ≥ 0) verändern das Optimum zusätzlich.
*Einschränkungen des Tests:* λ aus nur 800 Patienten; beim Annealer hinkt das Modell β hinterher, das gemessene β/d ist
daher eher eine Obergrenze; die Theorie ist nur für den linearen VAE bewiesen.

### A3. Vorhersage aus einem Snapshot (ohne Sampling oder Klassifikator)

| Größe | Ansatz | Quelle / Status |
|---|---|---|
| AUC auf z | Binormales Modell: zwei Klassen mit gemeinsamer Kovarianz Σ_w + σ²I; AUC = Φ(δ/√2), δ² = Δμᵀ(Σ_w + σ²I)⁻¹Δμ. Erklärt, warum z-AUC vor μ-AUC fällt (σ² ≈ 1 addiert sich zur Klassenstreuung) | Binormales ROC-Modell (z. B. Pepe 2003, *The Statistical Evaluation of Medical Tests*); Mehrklassen-Erweiterung über Paare (OvO wie unsere AUC) noch herzuleiten; nicht getestet |
| E[T(z)] | Monte Carlo aus μ, σ (heute 2 Züge); analytisch über Nachbarschafts-Wahrscheinlichkeiten unter Gauß-Rauschen | gemessen: E[T(z)] gegen log Informationsschranke R² ≈ 0,96 (39 Läufe, beschreibend); geschlossene Form offen |
| Clusterzahl | Moden von q(z) = 1/N Σ N(μ_i, σ²) (Gauß-KDE mit Bandbreite σ), Mean Shift | hergeleitet (q(z) ist exakt eine KDE); Silverman 1981, Fasy et al. 2014 für Signifikanz; nicht gebaut |
| T(μ) | nicht analytisch: rangbasiert und skaleninvariant | gemessen: T(0,01·μ) = T(μ) |

---

## Teil B: Roadmap (Stand 03.11.2026)

**Ziel 1 (jetzt):** maximal komprimieren ohne Posterior Collapse. **Ziel 2 (danach):** label-freie relative
Erhaltungsniveaus (Preservation, Clusterstruktur, Sparsity) als Sollwerte; Bewertung mit Labels nur am Ende
(Tabelle „Kriterium × Niveau → Downstream-AUC“). Ergebnisse im Detail: `checklist.md`.

### B0. Leitbild: Priors guter Repräsentationen (Bengio, Courville & Vincent 2013, §3.1) → Metriken

Die Priors sind die inhaltliche Quelle der Metriken: Jede Metrik prüft, ob eine Eigenschaft, die eine gute
Repräsentation laut Bengio et al. haben sollte, unter Kompression (steigendem β) erhalten bleibt. Volltext der Priors in
`checklist.md`. Spalte „Rolle“ wie in B1; „Vorschlag“ = noch nicht umgesetzt.

| Prior (Bengio et al. 2013) | Kernaussage | unsere Metrik | Rolle | Stand |
|---|---|---|---|---|
| **Smoothness** | x ≈ y ⇒ f(x) ≈ f(y) | Continuity, MRRE missing, R_NX (verlorene Nachbarn); Trustworthiness / MRRE false (falsche Nachbarn) | Preservation | gemessen: lokal bricht vor global; T(μ) unempfindlich, auf z verlieren alle Metriken stark |
| **Multiple explanatory factors** | mehrere unabhängige Faktoren, verteilte Repräsentation | η² pro Dimension (Krebsart, Geschlecht), D_eff, aktive Dims pro Patient; Total Correlation | Bewertung / Diagnose | gemessen: dim 0 trägt Geschlecht (η² ≈ 0,44) und Krebsart (≈ 0,82) und stirbt zuerst; D_eff flach bis zur Kante. TC ist MI-artig → nur beschreibend |
| **Hierarchical organization** | abstrakte Konzepte aus weniger abstrakten | AUC 33 Krebsarten gegen 8 Gruppen; Merge-Reihenfolge im ToMATo-Baum | Bewertung | gemessen: AUC 33 / 8 Klassen fallen gleich (0,99/0,98 → 0,79/0,78). **Vorschlag:** prüfen, ob ToMATo-Gipfel in der Reihenfolge der Label-Hierarchie verschmelzen (COAD/READ, LUAD/LUSC zuerst) |
| **Semi-supervised learning** | was P(X) erklärt, hilft für P(Y \| X) | Downstream-AUC auf z (logistische Probe) | **nur Bewertung** | gemessen: Krebs-AUC bleibt ≥ 0,96 bis Gini ≈ 0,75; Labels steuern nichts |
| **Shared factors across tasks** | Faktoren über Aufgaben geteilt | AUC über mehrere Aufgaben (`snapshot_tasks`: Krebsart, Gruppe, Histologie, Subtyp, Geschlecht, Stadium, Alter) | Bewertung | gemessen nur Krebsart und Geschlecht: Geschlecht geht zuerst (Halbwertszeit ≈ 0,06), Krebsart spät (≈ 0,5). **Vorschlag:** Reihenfolge des Verlusts über alle Aufgaben (Checkliste Zeile 1) |
| **Manifolds** | Masse nahe niedrigdimensionaler Mannigfaltigkeiten | T, LCMC, R_NX(k), intrinsische Dimension (TwoNN, Gride), Betti-Kurven / Persistenz; Circle, Swiss Roll, Sphere, GUDHI als Kontrollen | Preservation / Diagnose | gemessen: R_NX k = 200 hält bis zum Collapse; Persistenz-Entropie H1 von μ 5,5 → 3,9–4,3. Betti-Kurven und intrinsische Dimension über β noch offen |
| **Natural clustering** | Klassen auf getrennten Mannigfaltigkeiten, Täler niedriger Dichte dazwischen | ToMATo (Prominenz, signifikante Gipfel), HDBSCAN, Insel-Kohäsion; Brücken/Conductance, Bhattacharyya pro Paar | nur loggen | gemessen: Gipfel verschwinden früh (Halbwertszeit 0,07), Krebsart bleibt linear lesbar → **dieser Prior trägt die Information unter Kompression nicht**. Brücken/Bhattacharyya noch offen |
| **Temporal and spatial coherence** | benachbarte Beobachtungen gleiche Kategorie, langsame Faktoren | (keine Zeitreihe in TCGA) Stabilität zwischen Snapshots als Analogon | – | nicht anwendbar auf die Daten; **Vorschlag:** Snapshot-zu-Snapshot-Stabilität der Nachbarschaften als Rauschmaß für Steuersignale (B5) |
| **Sparsity** | nur wenige Faktoren pro Beobachtung relevant, Merkmale oft 0 | Gini der μ-Matrix, Null-Anteil, aktive Dims pro Patient, Aktivitätsmuster | Sollwert-Kandidat (B8) | gemessen: steigt monoton mit β; bei Varix durch die ReLU-artige μ-Klemme erzwungen, auch ohne Datenstruktur (Circle/Swiss Roll schon bei β = 0 41–52 % Nullen) |
| **Simplicity of factor dependencies** | einfache, meist lineare Abhängigkeiten; lineare Prädiktoren obendrauf | lineare Probe (AUC), lineare Gauß-Theorie (Teil A) | Bewertung / Theorie | gemessen: lineare Lesbarkeit bleibt, wenn Cluster weg sind; Teil A sagt Schwellen 2λ/D aus pPCA voraus (Test offen, Phase 2.6 alt) |

**Lesart für die Arbeit:** Unter Rate-Druck geht zuerst verloren, was *Natural clustering* und *Smoothness* auf kleiner
Skala ausmacht, *Sparsity* nimmt zu, und *Simplicity* (lineare Lesbarkeit) hält am längsten. Das ist eine Beschreibung
der Messungen auf TCGA, d = 8; ob die Reihenfolge allgemein gilt, prüft der Large-Scale-Teil (B9).

### B1. Metriken und ihre Rolle im Annealer

| Metrik | misst | Rolle | Stand / Begründung |
|---|---|---|---|
| R_j pro Dimension, besser Varianzanteil ½·Var μ_j | trägt die Dimension Information? | **harter Collapse-Wächter** | gemessen: R_j ≡ ε-δ-KL ≡ Active Units; Gesamtrate taugt nicht (0,4 Nats war schon Collapse). Varianzanteil, weil ein konstanter μ-Versatz die Untergrenze ohne Information erfüllt (hergeleitet; wichtig für Free Bits) |
| E[T(z)], LCMC / R_NX auf z | Nachbarschaftserhalt | Preservation-Nebenbedingung (Ziel 2) | T(μ) ist rangbasiert und sieht den Collapse nicht (hergeleitet + gemessen); LCMC/R_NX bei kleinem k warnt früher als T |
| Gini der ganzen μ-Matrix, Null-Anteil | Kompressionsgrad (Nullen pro Patient) | möglicher Sollwert jenseits der Preservation | gemessen: monoton (Spearman 0,996), ruhig; Gini ≈ 0,96 − 0,17·log₁₀ Rate (R² 0,97) → fast redundant zur Rate |
| D_eff = (Σ R_j)²/Σ R_j² | Raten-Verteilung über Dimensionen | nur Collapse-Anzeiger | gemessen: flach 7,5–7,8 bis 5·10⁻⁴, springt erst beim Collapse |
| ToMATo-Marge auf gezogenem z | signifikante Dichtegipfel | nur loggen | monoton (−0,83…−0,98), aber Rauschen in der Stufe ≈ Sprung zwischen Stufen |
| SNR_j, Σ ½ log(1 + SNR_j) | Signal gegen Rauschen σ | Kandidat für den Wächter | E[T(z)] hängt fast monoton daran (Spearman ≥ 0,995). Gauß-MI-Schranke → nach Regel nur beschreibend, außer bewusst deklarierte Ausnahme |
| H0 bei 2σ, Betti-Kurven | Trennbarkeit in Rauscheinheiten, Topologie | loggen | H0: eine Komponente ab β/d ≈ 1,2·10⁻³; Betti-Kurven noch nicht gerechnet |
| AUC auf z | Downstream | **nur Bewertung** | Labels steuern nichts |

### B2. Was zuerst verloren geht (gemessen, TCGA, d = 8)

Feinstruktur → **Geschlecht** (Halbwertszeit ≈ 0,06 auf normierter β-Achse, 1 = β 0, 0 = Kante) ≈ **ToMATo-Cluster**
(≈ 0,07) → LCMC / R_NX bei kleinem k → **Krebsart-AUC** (≈ 0,5) → **T** (≈ 0,6–0,7) → **Dimensionen**. Globale
Distanzordnung (Spearman, R_NX k = 200) hält bis zum Collapse. Folgerung: Cluster ≠ Information.

### B3. Hoch- und Runterfahren

| Richtung | Stand |
|---|---|
| Hoch (Leiter, festes β) | gemessen: Kante bei festem β zwischen 10⁻³ und 2·10⁻³; auf der Leiter 2·10⁻³ noch ohne Collapse. Schrittweises Hochfahren schiebt den Collapse hinaus: **Vermutung** |
| Halten (H/F, 2.000 Epochen) | gemessen: Gleichgewicht; T(z) 0,66–0,71 bei T(μ) = 0,9 |
| Runter (Umschaltversuch T(μ) 0,75 → 0,9, §13, fertig) | gemessen, 3 Seeds: T(μ) nach 80–95 Epochen wieder im Band, aber **von 10 toten Dimensionen kommen nur 2 zurück** (alle s17), obwohl β kurz auf ≈ 5·10⁻⁵ fällt. **Hysterese:** dasselbe T braucht 1,3–2,7× weniger β (4,2–8,2·10⁻⁴ statt 1,1·10⁻³), Rate höher (4,1–7,1 statt 3,0–3,5 Nats). Downstream kommt zurück (Krebs-AUC 0,84–0,85 gegen 0,82–0,84 direkt) |
| Treppe runter mit Abzweigen (§14, Seed 17, fertig) | gemessen: lebende Dims 8/7/3/2/0 bei Sollwert 0,95/0,85/0,75/0,65/0,55. Comeback auf T(μ) = 0,9 ab 3 und ab 2 lebenden Dims (je 4 Dims am Ende, Rate 8,5–9,1, Krebs-AUC 0,87); **ab vollständigem Collapse nicht** (T 0,77, Rate 10⁻³, AUC 0,52 nach 1.000 Epochen bei β = 0). Vollständiger Collapse aber nicht strikt absorbierend: nach 400 Epochen μ ≡ 0 wieder μ > 0 bei 92 % der Patienten (Ausweg über Weight Decay: **Hypothese**) |
| Offen | Hysterese-Kurve (Metrik gegen β, hin und zurück), β gezielt über die Kante und zurück ohne Regler, Regelung auf T(z), andere Datensätze und d |

Folge für das Design (gemessen, TCGA): **Teil-Collapse** ist für T(μ) umkehrbar (bis 2 lebende Dims), die toten Dimensionen aber meist nicht; **vollständiger Collapse** ist praktisch ein Point of no Return (innerhalb von 1.000 Epochen bei β = 0), wenn auch nicht strikt. Überschießen über die Kante kostet Dimensionen dauerhaft; der Wächter muss **vor** der Kante anhalten. Frühwarnung (LCMC kleines k, SNR, Rate pro Dimension dicht gemessen) wird damit zentral.

### B4. Collapse finden und pro Datensatz definieren

1. **Definition:** Dimension j kollabiert, wenn ihr Varianzanteil der Rate für ≥ 99 % der Samples unter ε liegt
   (ε-δ nach Lucas et al. 2019), bezogen auf die am Warm-up-Ende aktiven Dimensionen. Klemm-tote Dimensionen getrennt.
2. **ε label-frei kalibrieren:** Lücke in der Verteilung von log R_j über Dimensionen und Snapshots suchen (TCGA:
   tote ≤ 9 aktive Patienten, lebende ≥ 17, robust über ε 10⁻⁴…10⁻¹). Keine Lücke ist selbst ein Befund.
3. **Rest-Rate:** Varix-Boden ≈ 2·10⁻⁴ Nats (logvar-Klemme). „Rate 0“ gibt es nicht; bei vollständigem Collapse ist der
   Gini undefiniert, nicht 1.
4. **Auflösung:** Der Sprung passiert in < 20 Epochen → Wächter nahe der Kante alle 5 Epochen messen.

**Zwei Collapse-Grenzen — die ersten Boundaries für TCGA und die Geometrie-Datensätze.** Beide werden gemessen und
berichtet, weil die Literatur und wir Verschiedenes meinen:

| Grenze | Definition | Bezug |
|---|---|---|
| **β_partial** (Teil-Collapse) | kleinstes β, bei dem mehr Dimensionen tot sind als bei β = 0 im selben Seed (ε-δ, ε = 0,01 Nats, δ = 1 %) | unser Ziel: die erste Dimension stirbt, andere laufen weiter |
| **β_total** (vollständiger Collapse) | kleinstes β, bei dem alle d Dimensionen tot sind (Rate am Boden ≈ 2,5·10⁻⁵ Nats pro Dimension) | ControlVAE / Kapazität C: KL → 0, der Decoder ignoriert z ganz |

Erste Messung aus der 106er-Studie (festes β, Epoche 750; die Raster sind grob, angegeben ist die erste Stufe mit dem
Ereignis, davor die letzte ohne):

| Datensatz, d | Seeds | β_partial (β/d) | β_total (β/d) |
|---|---|---|---|
| TCGA, 8 | 13 / 17 / 23 / 31 | 2·10⁻³ / 5·10⁻⁴* / 2·10⁻³ / 2·10⁻³ (letzte Stufe ohne: 10⁻³) | 8·10⁻³ / 4·10⁻³ / 4·10⁻³ / 4·10⁻³ |
| Circle, 2 | 13 / 17 | 5·10⁻³ / 3·10⁻² | 2·10⁻² / 3·10⁻² |
| Swiss Roll, 2 | 13 / 17 | 10⁻³ / 2·10⁻² | 5·10⁻³ / 2·10⁻² |

\* s17: eine Dimension ist bei 5·10⁻⁴ und 10⁻³ schon ab Epoche 50 tot, bei 2·10⁻³ bleibt es bei dieser einen; das sieht
nach Klemm-Tod aus, nicht nach β-Collapse. Bei festem β gibt es kein Warm-up, Klemm-Tod und β-Collapse lassen sich
deshalb nicht sauber trennen.

Lesart (gemessen, grob): Auf TCGA liegen beide Grenzen einen Faktor 2–4 auseinander; zwischen ihnen kollabiert das Modell
teilweise (bei 2·10⁻³ 1–6 tote Dimensionen, AUC 0,61–0,67), ohne dass die Gesamt-KL auf 0 geht. Genau diesen Bereich
sieht eine Regelung auf die Gesamt-KL nicht. Bei d = 2 sind die Synthetik-Werte unruhig (Seed 17 hat schon bei β = 0 eine
klemm-tote Dimension, Swiss Roll s17 springt zwischen 0 und 1 toten Dimensionen) und die Grenzen streuen über eine
Größenordnung zwischen den Seeds. **Nächster Schritt:** feinere β-Raster um beide Grenzen, mehr Seeds, d ≥ 2 bei den
synthetischen Daten mit Warm-up (dann lassen sich klemm-tote Dimensionen vorher abziehen), GUDHI-Datensätze.

### B5. Engineering: Korrelationen systematisch finden (offline, kein Training)

Tabelle über alle Snapshots (106er-Studie, Leiter, Block A/B/C/H/F/S, Large Scale). Pro Metrik:
Spearman mit log Rate und β; **partielle** Korrelation mit AUC bei festgehaltener log Rate (Information über die Rate
hinaus); Rauschen in der Stufe gegen Sprung zwischen Stufen; **Vorlaufzeit** vor der ersten toten Dimension (Epochen,
β_rel); Redundanz (Korrelationsmatrix, Gruppen); Stabilität über Seeds, k, d, Datensätze. Snapshots sind abhängig →
Kennzahlen pro Lauf oder Block-Bootstrap über Läufe, keine naiven p-Werte.

Bekannt (gemessen): Gini ↔ log Rate R² 0,97; E[T(z)] ↔ SNR-Schranke ≥ 0,995; Null-Anteil ↔ β 0,99;
ToMATo −0,83…−0,98; T(μ) blind für Collapse; D_eff nur am Collapse.

### B6. β senken und Preservation zurückgewinnen

Umschaltversuch (§13) hat es für TCGA beantwortet: Preservation ja, Dimensionen meist nicht (2 von 10). Treppe (§14) ergänzt: Comeback noch ab 2 lebenden Dimensionen (1 bzw. 2 Dims kehren zurück), ab 0 nicht mehr (Seed 17). Danach: gleiche Frage auf mindestens einem synthetischen Datensatz und
mit β-Exkursion ohne Regler (fester Plan hoch → runter), damit Regler und Modell getrennt sind.

### B7. Nutzung durch andere (Nutzer, Agents, AUTOENCODIX-PR)

- **Standard:** nur Collapse-Wächter → maximal komprimieren ohne Collapse (Ziel der Arbeit).
- **Steckbar:** `Constraint(name, measure, target, tolerance)` im Maximin (T(z), Gini-Sollwert, Cluster-K); geloggt
  wird die bindende Bedingung.
- **AUC-Maximierung durch Nutzer:** überwachte Nebenbedingung oder β-Wahl nach Validierungs-AUC als Hook, klar
  deklariert; die Arbeit selbst bleibt label-frei.
- **Agent-Schnittstelle:** start / status / report. Rechenkosten pro Metrik dokumentieren (T auf 800 Samples alle
  5 Epochen billig, ToMATo ≈ 0,6 s).

### B8. Raumaufteilung numerisch für β (jenseits der Preservation)

Gini / Null-Anteil / aktive Dims pro Patient als Sollwert: nutzbar etwa Gini 0,44–0,88, anschaulich, aber praktisch ein
Raten-Sollwert (wie Kapazität C, Burgess et al. 2018); träge bei kleinem β; Umkehrbarkeit offen (§13). D_eff ungeeignet
(flach). **Zuerst offline:** Kalibrierkurve Gini ↔ β pro Seed aus der 106er-Studie; trifft festes β den Ziel-Gini schon,
braucht es keinen Regler.

### B9. Large-Scale-Setup: mit und ohne Annealing

**Vorhanden:**

| Suite | Annealing | Aufbau | Stand |
|---|---|---|---|
| `fixed_beta_106_epochs750_2026_10_27` | **ohne** (`no-annealing`, β fest) | TCGA d = 8: 15 β/d (0 … 1,5·10⁻²) × Seeds 13/17/23/31; Circle und Swiss Roll d = 2: 11 β/d (0 … 0,1) × Seeds 13/17; + 2 TCGA β = 0 ohne Weight Decay; 750 Epochen, Split-Seed 2, Snapshots alle 50 (Train/Valid/Test) | **106/106 fertig**; ausgewertet in `gini_index.ipynb` (Sparsity, T(z), AUC) |
| `large_scale_core_2026_10_26` | **fester Plan** (lineares Warm-up 300 Epochen, dann β konstant; `3phase-linear` + `beta_schedule`) | TCGA, Circle, Sphere, Swiss Roll (+ correlated, + noisy copies); d ∈ {4, 8, 16} (Circle 2); 2 β/d je Datensatz (TCGA 10⁻³ / 1,5·10⁻²; synthetisch 0,01 / 1); Seeds 13/17/23/31; Split-Seeds 2/3/5; 1.000 Epochen; Bestätigung auf Split 7 / Daten-Seed 73 vorgemerkt | **nur Circle (3 × 8) und Sphere Split 2 (19) fertig**; Rest nicht gelaufen |
| `annealing_recovery_design` (14./20.10.) | fester Zyklus niedrig → hoch → niedrig | 648 Läufe entworfen | nur CPU-Smoke-Test |
| Prototyp-Annealer (`trustworthiness_annealer`) | **Regler** (Maximin auf T(μ)) | TCGA d = 8, 3 Seeds, Blöcke A/B/C/H/F/L/R/S | fertig bis auf §13 (läuft) |

**Achtung:** `beta_schedule` und `beta_controller` sind in `new_beta_varix_experiments.py` inzwischen abgeklemmt.
`large_scale_core` und der Recovery-Entwurf laufen deshalb nicht mehr über die Pipeline; dort geht nur noch
`no-annealing` oder die nativen AUTOENCODIX-Kurven (`logistic-mid` …). Der Regler läuft als Loss-Unterklasse im
Notebook bzw. später im Annealer-Modul.

**Vorschlag für den Vergleich mit und ohne Annealing** (braucht Training, vorher bestätigen):

| Arm | β-Steuerung | Zweck |
|---|---|---|
| F (fest) | β/d-Raster, `no-annealing` | Referenzkurve, Kante β_edge pro Datensatz und d (106er-Studie erweitern) |
| N (nativ) | AUTOENCODIX `logistic-mid` bis β_max | was Nutzer heute bekommen |
| W (Wächter) | Maximin nur mit Collapse-Wächter (Varianzanteil R_j ≥ ε) | Ziel 1: erreicht er β ≈ β_edge (Faktor ≤ 1,2) ohne Collapse und ohne Überschießen? |
| W+T | Wächter + E[T(z)] ≥ T_min | Ziel 2, bindende Bedingung loggen |
| FB | Free Bits (VarixLoss-Unterklasse) | Pflicht-Baseline |
| Z | zyklisches Annealing (Fu et al. 2019) | Baseline |
| KL | PI-Regler auf die **Gesamt-KL** mit Sollwert (ControlVAE, Shao et al. 2020; Kapazität C, Burgess et al. 2018) | Baseline aus der Literatur: verhindert β_total, aber nicht β_partial? (direkter Test) |

**Nachbarschaftsgröße k ∈ {10, 20, 40}** für alle k-abhängigen Größen: im Arm W+T als Regler-k (eigener Lauf je k),
in allen Armen als Messung (T, LCMC/R_NX, ToMATo-kNN-Graph). Bisher gemessen nur in Block B (TCGA, Ziel T(μ) = 0,9):
k = 10/20/40 → Rate 3,2–3,8 Nats ohne Collapse; k = 200 erreicht das absolute Ziel nie. Deshalb mit **relativen**
Sollwerten (Anteil des Warm-up-Werts bei demselben k), sonst ist der Vergleich über k nicht fair. k = 40 ≈ 5 % von
n = 800 (Controller-Subset); bei den synthetischen Daten k relativ zu n prüfen.

Datensätze: TCGA, Circle, Swiss Roll, Sphere, GUDHI Torus/Spirale/Bunny/Activities (Lizenzen geprüft: Activities
CC BY 4.0, Bunny Stanford-Bedingungen nur Forschung + Quellenangabe, Spirale MIT, Torus generiert). d ∈ {2, 4, 8, 16},
d ≤ Eingangsdimension bei den synthetischen Daten. Seeds 13/17/23/31, Split-Seed 2 für die Entwicklung, Split 7 /
Daten-Seed 73 nur für die Abschlussbestätigung. Snapshots alle 50 Epochen, nahe der Kante dichter.
Ressourcen: RAM ≈ 64 MB pro Checkpoint (TCGA) → höchstens ≈ 100 Checkpoints pro Lauf; ein Lauf zur Zeit.

### B10. Reihenfolge

1. **Erledigt:** Umschaltversuch §13 (3 Seeds): Collapse weitgehend nicht umkehrbar, Hysterese im β.
2. **Offline, sofort:** Korrelationstabelle (B5) inkl. Vorlaufzeit; ToMATo und Collapse auf der 106er-Studie inkl.
   Circle/Swiss Roll; Gini-Kalibrierkurve (B8); ε-Lücke pro Datensatz (B4).
3. **Training, kurz:** feine Leiter / festes β zwischen 10⁻³ und 4·10⁻³ (β_edge genauer); β über die Kante und zurück.
4. **Training, groß:** Arme F/N/W/FB (B9) auf TCGA + 2 synthetischen Datensätzen, dann W+T (k = 10/20/40, relative
   Sollwerte) und Z, dann weitere Datensätze und d.
5. **Bestätigung** auf Split 7 / Daten-Seed 73.
6. **Schreiben:** Einordnung über Bengio et al. 2013 (Priors: Sparsity, Natural clustering, Manifolds); Literaturlücke
   per Zitationssuche absichern (Wu & Fischer 2020, Ichikawa & Hukushima 2024).

### B11. Noch nicht abgedeckt / Entscheidungen

1. Baselines Free Bits, zyklisch, festes β knapp unter der Kante sind Pflicht; bisher war festes β so gut wie der Regler.
2. Abhängigkeit von d: Schwellen hängen stark an d (OntixTutorial-Sweep).
3. Klemme als Störfaktor: Klemm-Tod gegen β-Collapse trennen; mit den Professoren klären, ob die μ-Klemme gewollt ist.
4. MI-Regel: SNR-Schranke als Wächter nur mit deklarierter Ausnahme.
5. T auf μ oder z im Regler: für Ziel 1/2 z (Briefing); μ nur für Vergleich mit Block A und für den Umschaltversuch.
6. Test-Split erst ganz am Ende, nichts darauf abstimmen.

### Erledigt (Kurzfassung, Details in `checklist.md`)

Maximin-Annealer (identisch zum alten Controller); 39 Läufe Block A/B/C; SNR-Zerlegung; Collapse-Bestimmung robust;
Haltetest H/F; Plateau-Leiter; konvergierte Referenzmodelle; ToMATo (`tomato_ladder`); Sparsity (`sparsity_ladder`);
Gini über 106 Läufe (`gini_index`); Lizenzen GUDHI-Daten.

---

igenes Analyse-Notebook?

## Quellen

- Burda, Grosse & Salakhutdinov 2016, *Importance Weighted Autoencoders*, ICLR (Active Units, §5.2).
- Cover & Thomas 2006, *Elements of Information Theory*, 2. Aufl., Thm. 8.6.5 (maximale Entropie), Thm. 10.3.3 (Reverse Water-Filling).
- Hoffman & Johnson 2016, *ELBO surgery*, NeurIPS Workshop (Zerlegung der Rate, Gl. 13–16).
- Lucas, Tucker, Grosse & Norouzi 2019, *Don't Blame the ELBO! A Linear VAE Perspective on Posterior Collapse*, NeurIPS (linearer VAE = pPCA; (ε, δ)-Collapse §5).
- Tipping & Bishop 1999, *Probabilistic Principal Component Analysis*, JRSS B.
- Pepe 2003, *The Statistical Evaluation of Medical Tests for Classification and Prediction* (binormales ROC-Modell).
- Silverman 1981, *Using Kernel Density Estimates to Investigate Multimodality*, JRSS B; Fasy et al. 2014, *Confidence Sets for Persistence Diagrams*, Ann. Statist.
- Alemi et al. 2018, *Fixing a Broken ELBO*, ICML; Cremer, Li & Duvenaud 2018, *Inference Suboptimality in VAEs*, ICML.
- Bengio, Courville & Vincent 2013, *Representation Learning: A Review and New Perspectives*, IEEE TPAMI (§3.1 Priors).
- Hurley & Rickard 2009, *Comparing Measures of Sparsity*, IEEE Trans. Inf. Theory (Gini).
- Chazal, Guibas, Oudot & Skraba 2013, *Persistence-Based Clustering in Riemannian Manifolds*, J. ACM (ToMATo).
- Kingma et al. 2016, *Improved Variational Inference with Inverse Autoregressive Flow*, NeurIPS (Free Bits).
- Fu et al. 2019, *Cyclical Annealing Schedule*, NAACL; Burgess et al. 2018, *Understanding disentangling in β-VAE* (Kapazität C).
- Barshan & Altun 2010, *Daily and Sports Activities* [Dataset], UCI ML Repository, doi:10.24432/C5C59F (CC BY 4.0).
- Stanford 3D Scanning Repository (Bunny), Nutzung für Forschung mit Quellenangabe, kein kommerzieller Gebrauch.
- Shao et al. 2020, *ControlVAE: Controllable Variational Autoencoder*, ICML (PI-Regler auf die KL, arXiv 2004.05988).
- McCarthy, Li, Gu & Dong, *Addressing Posterior Collapse with Mutual Information for Improved Variational Neural Machine Translation* (Johns Hopkins / Facebook; vermutlich ACL 2020, Ort nicht geprüft). Nur Quelle zu Posterior Collapse: Collapse = totaler Collapse (starker Decoder ignoriert z), Abhilfe über ein geändertes Ziel (MI-Term zurück ins ELBO nach Hoffman & Johnson 2016, Bag-of-Words-Hilfsdecoder); nicht implementiert, keine Baseline.
