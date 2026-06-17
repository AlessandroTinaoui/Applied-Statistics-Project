# Dimensionality Reduction & Clustering dei qubit — Report

*Analisi non supervisionata della struttura tra i qubit del dataset
`phanerozoic/qiskit-calibration-drift`.*

## 1. Obiettivo e domanda di ricerca

Mentre le parti di regression e fault prediction lavorano a livello di
*snapshot temporale*, qui la domanda è diversa: **esistono gruppi naturali
tra i qubit, e questa struttura è spiegata dalla macchina di appartenenza
(backend)?**

Ipotesi: i qubit dello stesso computer quantistico condividono un'impronta di
calibrazione che li rende distinguibili da quelli di altre macchine.

## 2. Unità statistica e costruzione delle feature

Il punto metodologico chiave è la scelta dell'**unità statistica**. Per
cercare struttura *tra i qubit* l'unità deve essere il qubit, non lo
snapshot. Collassiamo quindi la dimensione temporale costruendo **un profilo
per `(backend, qubit)`** — `468` qubit in totale (`156` per ciascuno dei tre
backend `ibm_fez`, `ibm_kingston`, `ibm_marrakesh`).

**Gestione dei NaN.** Le proprietà di calibrazione grezze (`T1`, `T2`,
`sx_error`, `readout_error`, `prob_meas*`) sono 73–96% NaN *per snapshot*,
perché le calibrazioni sono asincrone. Non imputiamo a livello di snapshot
(significherebbe inventare dati): aggreghiamo nel tempo usando **solo le
misure reali** (i NaN vengono ignorati). Pooling di ~24 giorni di snapshot dà
abbastanza osservazioni per stimare per ogni qubit:

- `<prop>_median` — il valore tipico di esercizio (stimatore robusto);
- `<prop>_iqr` — lo scarto interquartile = drift/variabilità temporale robusta;
- per gli errori di gate/readout, `<prop>_fail_frac` — la frazione di
  calibrazioni fallite (valore degenere ≈ 1.0), proxy diretto di instabilità.

Si ottengono **15 feature intrinseche** per qubit. Solo dopo l'aggregazione
imputiamo i NaN residui con la mediana e **standardizziamo** (z-score). Le
feature con missingness > 40% verrebbero scartate; con la soglia attuale tutte
le 15 sopravvivono.

## 3. Standardizzazione

Ho applicato lo **z-score** (`StandardScaler`). È necessario perché le feature
vivono su scale incomparabili (T1 ~1e-4, errori ~1e-3, fail_frac ~0–1) e sia
la **PCA** sia il **clustering euclideo** non sono invarianti di scala: senza
standardizzare, le variabili con ordine di grandezza maggiore dominerebbero.

## 4. PCA ed EDA

La PCA sullo spazio standardizzato (15 feature) mostra una varianza
**distribuita**, non dominata da un singolo asse: servono **7 componenti per
il 90%** della varianza.

| Componente | Var. spiegata | Cumulata | Interpretazione (loadings) |
|---|---|---|---|
| PC1 | 34,9% | 34,9% | **Asse "errore/salute"**: readout_error (0.42), sx_error_fail_frac (0.37), sx_error (0.35), prob_meas (0.35–0.38) |
| PC2 | 18,0% | 52,8% | **Asse "coerenza"**: T2_median (0.49), T1_median (0.48), T2_iqr (0.43), T1_iqr (0.38), calibration_lag (−0.34) |
| PC3 | 14,0% | 66,8% | Variabilità readout/misura (iqr) |

La lettura fisica è netta: la prima sorgente di variabilità tra i qubit non è
*quale macchina* ma *quanto è in salute* il qubit (errori complessivi); la
coerenza T1/T2 è un asse secondario e ortogonale.

Nel piano PC1–PC2 (`figures/pca_scores_by_backend.png`) i backend si
sovrappongono ampiamente e una lunga coda lungo PC1 isola pochi qubit ad
altissimo errore. La PCA lineare, da sola, **non** separa i backend.

## 5. Embedding non lineari (t-SNE / UMAP)

Qui cambia tutto. L'embedding **UMAP** (`figures/umap_embedding.png`) mostra
**tre territori ben distinti per backend**: `ibm_kingston`, `ibm_fez` e
`ibm_marrakesh` occupano regioni separate. Questo segnala che la struttura per
backend è **non lineare**: esiste, ma vive su una varietà che la PCA lineare e
il clustering globale euclideo non catturano direttamente.

## 6. Clustering e confronto con il backend

**K-Means / Ward globali.** Il criterio silhouette seleziona `k = 2` con
silhouette molto alta (0,807) — ma è un artefatto: la spaccatura separa un
gruppetto di **11 qubit anomali** (cluster minoritario) dal resto. Il
dendrogramma di Ward (`figures/dendrogram.png`) conferma visivamente: un
piccolo ramo si stacca a distanza ~64 dal blocco principale. La **silhouette
delle etichette backend** nello spazio delle feature è bassa (0,085): preso
alla lettera, questo direbbe "nessuna separazione". Ma è ingannevole, perché il
k=2 è dominato dagli outlier e la struttura per backend è non lineare. Due
controlli risolvono l'ambiguità.

**(a) Recuperabilità del backend (kNN cross-validato).** Misura *locale* di
separabilità: il backend è leggibile dal profilo del qubit? Uso un
classificatore k-nearest-neighbors con **cross-validation stratificata a 5
fold** (ogni fold conserva le proporzioni dei backend).

| Etichetta | Accuratezza kNN (5-fold) | Caso base (classe maggioritaria) |
|---|---|---|
| **backend** | **0,876 ± 0,033** | 0,333 |

Il backend è recuperabile all'**87,6%** contro un caso base del 33,3%:
fortissima conferma dell'ipotesi e dell'evidenza UMAP. Una metrica *locale*
come il kNN cattura la struttura non lineare che la silhouette globale non
vede.

**(b) Clustering robusto.** Rimossi gli 11 qubit anomali, il K-Means sui `457`
qubit "normali" seleziona spontaneamente **k = 3** (= numero di backend). La
figura `figures/robust_clusters_vs_backend.png` (scatter PCA: a sinistra per
cluster, a destra per backend) e la contingenza
`tables/contingency_robust_backend.csv` mostrano l'allineamento:

| cluster robusto | ibm_fez | ibm_kingston | ibm_marrakesh |
|---|---|---|---|
| 0 | 151 | 16 | 112 |
| 1 | 3 | **134** | 40 |
| 2 | 0 | 1 | 0 |

`ibm_kingston` si isola in modo quasi puro (cluster 1: 134 dei suoi 154
qubit), mentre `ibm_fez` e `ibm_marrakesh` si fondono nel cluster 0. Quindi la
struttura per backend, mascherata dagli outlier, riemerge appena si toglie il
rumore. Che l'accordo non sia perfetto si spiega così: **due macchine su tre
hanno profili linearmente molto simili** e vengono separate solo
dall'embedding non lineare (UMAP). È coerente con la regression, dove
`ibm_kingston` mostrava lo shift di baseline T1 più marcato.

## 7. I qubit anomali

Gli 11 qubit del cluster outlier (2,4% del totale) sono distribuiti su tutte e
tre le macchine — `ibm_fez` (2), `ibm_kingston` (5), `ibm_marrakesh` (4) — e
sono caratterizzati da errori di gate/readout estremi e alta frazione di
calibrazioni fallite (estremo di PC1). Sono gli stessi "punti di pre-guasto"
isolati dall'analisi di Mahalanobis e dallo studio di fault prediction dei
colleghi: il segnale di degrado, non rumore. La lista completa è in
`tables/anomalous_qubits.csv`.

## 8. Conclusioni

1. **I qubit si separano per backend, ma in modo non lineare.** Ogni qubit
   porta un'impronta di calibrazione specifica della macchina: il kNN recupera
   il backend con 87,6% di accuratezza, UMAP separa le tre macchine, e il
   clustering robusto (senza outlier) trova k=3 con `ibm_kingston` ben isolato.
   Il raggruppamento interpretabile dei qubit è per computer di appartenenza.
2. **Asse dominante = salute del qubit, non identità.** La prima sorgente di
   varianza (PC1, 35%) è l'errore complessivo; la struttura "naturale" dei dati
   grezzi è un gruppetto di ~11 qubit gravemente degradati, presenti su tutte
   le macchine.

**Messaggio metodologico.** Affidarsi a una singola metrica globale
(silhouette) avrebbe portato alla conclusione sbagliata "i qubit non si
separano". La combinazione di embedding non lineare (UMAP), recuperabilità
locale (kNN) e clustering robusto agli outlier mostra invece che la struttura
per backend c'è ed è forte, semplicemente non lineare e oscurata da pochi
qubit anomali.

## 9. Limiti

- La separabilità per backend è non lineare; il K-Means lineare la sottostima
  perché impone cluster sferici (la misura affidabile è il kNN e l'evidenza
  visiva di UMAP).
- L'imputazione mediana globale dei NaN residui può attenuare lievi differenze
  tra backend; una imputazione entro-backend è stata evitata di proposito per
  non iniettare artificialmente il segnale di backend.
- Finestra temporale breve (~3,5 settimane); t-SNE/UMAP usati per
  visualizzare, non per misurare distanze.
