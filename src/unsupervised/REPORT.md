# Dimensionality Reduction & Clustering dei qubit — Report

*Analisi non supervisionata della struttura tra i qubit del dataset
`phanerozoic/qiskit-calibration-drift`.*

## 1. Obiettivo e domanda di ricerca

Domanda puramente **non supervisionata**: **esistono strutture consistenti tra
i qubit?** Tutta l'analisi (PCA, embedding, clustering) è condotta **senza
etichette**. Solo alla fine (§8) introduciamo il `backend` (la macchina) come
**chiave di lettura a posteriori**, per interpretare le strutture emerse — mai
per addestrare un modello.

## 2. Unità statistica e costruzione delle feature

L'unità statistica è il **qubit**, non lo snapshot: ogni qubit è una serie
temporale di calibrazioni, che collassiamo in **un profilo per `(backend,
qubit)`** — `468` qubit (`156` × 3 backend). Per ogni proprietà aggreghiamo nel
tempo con **mediana** (valore tipico) e **IQR** (drift), più **fail_frac** (per
gli errori) — statistiche robuste agli spike di calibrazione fallita.

**Selezione delle feature guidata dalla matrice di correlazione.** Si ottengono
**9 feature**: `T1_median/iqr`, `T2_median/iqr`, `readout_error_median/iqr`,
`sx_error_fail_frac`, `readout_error_fail_frac`, `calibration_lag_median`.
Rispetto al set grezzo abbiamo tolto:

- le famiglie **`prob_meas0/1`**, ridondanti (`readout_error` ≈ media delle due
  → blocco di lettura "contato tre volte");
- **`sx_error_median`/`sx_error_iqr`**: `sx_error` è **bimodale** (~0 nel regime
  normale, esattamente 1.0 sulle calibrazioni fallite), quindi median/IQR sono
  distorte dagli spike; teniamo solo `sx_error_fail_frac`,
  che è la sintesi numerica pulita del regime di fallimento.

**Gestione dei NaN.** Le proprietà grezze sono 73–96% NaN *per snapshot*: non
imputiamo lì, aggreghiamo sulle sole misure reali. I NaN residui del profilo
sono imputati con la mediana.

## 3. Standardizzazione

**z-score** (`StandardScaler`): le feature vivono su scale incomparabili e sia
la PCA sia il clustering euclideo non sono invarianti di scala.

## 4. PCA ed EDA — con outlier

La PCA mostra varianza **bilanciata** (6 componenti per il 90%):

| Componente | Var. | Interpretazione (loadings) |
|---|---|---|
| PC1 | 30,7% | **Coerenza / qualità**: T1_median 0.45, T2_median 0.43, T1/T2_iqr ~0.40 (+); readout_error −0.29, sx_error_fail_frac −0.27, calibration_lag −0.30 |
| PC2 | 25,9% | **Errori**: readout_error 0.53, sx_error_fail_frac 0.47, readout_error_fail_frac 0.45 |

I due assi interpretabili — **coerenza (PC1)** ed **errori (PC2)** — sono ora
distinti e netti. Nel piano PC1–PC2 (`figures/pca_scores.png`) la maggioranza
dei qubit forma un blocco, con una **coda di pochi qubit ad altissimo errore**
(outlier) lungo PC2.

## 5. Embedding non lineari (UMAP / t-SNE)

L'UMAP (`figures/umap_embedding.png`) rivela **regioni ben separate** → la
struttura è in parte **non lineare**. Usato solo per **visualizzare**: non
clusterizziamo su UMAP (distorce le distanze).

## 6. Clustering con outlier: DBSCAN

Con gli outlier presenti, il metodo giusto è **density-based**. **DBSCAN**
assegna ai qubit a bassa densità l'etichetta **noise** (−1) invece di forzarli
in un cluster: gestisce gli outlier **nativamente**, senza rimozione manuale e
senza creare singleton (a differenza del K-Means). Trova **2 cluster + 47 noise**
(`figures/dbscan_clusters.png`, `figures/dbscan_cluster_profiles.png`):

| cluster | n | T1 mediana | readout_error | lettura |
|---|---|---|---|---|
| 1 | 130 | ~272 µs | 0,012 | **alta coerenza**, errori bassi |
| 0 | 291 | ~171 µs | 0,018 | **coerenza più bassa** |
| noise | 47 | ~244 µs | **0,107** | qubit **degradati** (20% di fallimenti `sx_error`) = pre-guasti |

I due cluster si dividono per **coerenza** (T1/T2), non per errori; il **noise**
raccoglie i qubit degradati — gli stessi "pre-guasti" della fault prediction dei
colleghi (isolati con la distanza di Mahalanobis). Lista in
`tables/dbscan_cluster_means.csv`.

## 7. Rimozione degli outlier — PCA & UMAP

Mettendo da parte i qubit degradati, la **PCA resta sull'asse della coerenza**
(`figures/pca_loadings_clean.png`, `figures/pca_scores_clean.png`): PC1 = T1/T2
(T1_median 0.52, T2_median 0.49, T2_iqr 0.43, T1_iqr 0.41, calibration_lag
−0.37), e tra i qubit sani gli errori sono ≈ costanti (loadings ≈ 0). Bastano
**5 componenti** per il 90%. Anche l'UMAP pulito (`figures/umap_embedding_clean.png`)
mostra le stesse regioni più nitide.

## 8. Clustering senza outlier: DBSCAN vs K-Means

Tolti gli outlier, confrontiamo i due workflow.

- **K-Means (partizionale).** Forzato a `k=3`, dà gruppi 254/113/86 , ma **non ricostruisce
  bene le macchine**: `fez` si separa (149/154), mentre `kingston` è spaccato
  (83 + 56) e `marrakesh` è sparso — accordo col backend ~57%. Il K-Means impone
  cluster sferici e fatica su una struttura **non sferica a due livelli**.
- **DBSCAN (density-based).** Trova i **2 gruppi reali** in modo pulito
  (`figures/cluster_profiles.png` vs `figures/dbscan_cluster_profiles_clean.png`).

Messaggio metodologico: su questa struttura, il **partizionale (K-Means) fatica**
mentre il **density-based (DBSCAN) la cattura**. Il K-Means serve quindi come
**baseline di confronto** (mostra perché serve DBSCAN), non come soluzione.

## 9. Interpretazione a posteriori: il backend

Solo ora leggiamo i cluster DBSCAN con l'etichetta `backend`
(`tables/dbscan_contingency_backend.csv`):

| cluster DBSCAN | ibm_fez | ibm_kingston | ibm_marrakesh |
|---|---|---|---|
| noise (−1) | 5 | 26 | 16 |
| 0 | 151 | 0 | 140 |
| 1 | 0 | **130** | 0 |

`ibm_kingston` è isolato al **100%** (cluster 1, 130 qubit puri), `ibm_fez` e
`ibm_marrakesh` sono **fusi** nel cluster 0, gli anomali nel noise. Incrociando
con le medie, il cluster di kingston è quello a **T1/T2 più alti**: **kingston
è la macchina ad alta coerenza** — l'asse fisico (coerenza) e l'identità della
macchina coincidono, ed è lo stesso shift di baseline T1 visto dalla regression.

## 10. Conclusioni

1. **Struttura dominante = la "salute" del qubit.** L'asse principale (PC1) è la
   coerenza/qualità; la struttura più marcata dei dati grezzi è una **coda di
   qubit degradati** (i 47 punti di noise / gli ~11 più estremi), trasversale
   alle macchine — i "pre-guasti" della fault prediction.
2. **Struttura secondaria = la macchina, a due livelli.** I qubit sani si
   raggruppano in due gruppi di coerenza che corrispondono ai backend: **kingston
   distinto** (alta coerenza), **fez ≈ marrakesh**. È una struttura **non
   lineare e non sferica**: la cattura DBSCAN (e UMAP la mostra), il K-Means
   solo in parte.

**Messaggio metodologico.** Nessuna singola metrica o metodo basta: PCA +
embedding non lineare + confronto **K-Means vs DBSCAN**. DBSCAN è il metodo di
elezione (gestisce gli outlier, trova i gruppi reali); il K-Means è il termine
di paragone; gli outlier sono un *risultato* (i pre-guasti), non solo rumore.

## 11. Limiti

- La struttura per macchina è non lineare e a due livelli; il K-Means euclideo
  la recupera solo parzialmente.
- La finestra temporale è breve (~3,5 settimane); t-SNE/UMAP usati per
  visualizzare, non per misurare distanze.
- DBSCAN richiede di tarare `eps` (scelto via k-distance) ed è sensibile in alta
  dimensione; qui la struttura è abbastanza netta da renderlo stabile.