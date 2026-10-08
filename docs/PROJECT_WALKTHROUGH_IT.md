# Cable Instance Segmentation — Come raccontare il progetto

> Guida in italiano per spiegare **motivazione, system design, scelte tecniche e limiti** del progetto, realizzato insieme a Gianluigi Oricchio. Il riferimento è il notebook sperimentale completo, non una pipeline ridotta o riscritta.
>
> [System design tecnico e diagrammi Mermaid](SYSTEM_DESIGN.md) · [Notebook completo](../Segmentation.ipynb) · [README](../README.md)

## 1. Il problema che ha motivato il progetto

Il problema non era semplicemente riconoscere la presenza di un cavo elettrico in un'immagine. Volevamo esplorare la **segmentazione delle singole istanze di cavi nelle immagini aeree**, un compito difficile perché i cavi sono lunghi, estremamente sottili e spesso vicini o sovrapposti.

Una bounding box può includere soprattutto sfondo, e anche uno spostamento di pochi pixel può penalizzare molto la sovrapposizione tra due maschere. Inoltre, nelle immagini dense non basta individuare una struttura lineare: bisogna distinguere **quanti cavi ci sono e a quale istanza appartengono i pixel**.

La domanda progettuale centrale è:

**Quanto è efficace un modello di instance segmentation basato su Transformer su oggetti filamentari, e quali informazioni perdiamo valutandolo soltanto con metriche basate sull'overlap?**

Da questa domanda derivano i due filoni del lavoro: **segmentazione con RF-DETR** e **analisi geometrica delle maschere**.

## 2. La soluzione raccontata semplicemente

Il progetto segue tre macro-attività:

1. **Comprendere il dataset:** esaminare annotazioni COCO, maschere, bounding box, spessore, frammentazione e numero di cavi per immagine.
2. **Addestrare e valutare RF-DETR:** usare un modello di segmentazione a istanze per produrre maschere distinte, assegnare score e misurare la qualità con COCOeval.
3. **Studiare la geometria e gli errori:** utilizzare PCA e altri metodi sperimentali per capire la direzione dei cavi, testare possibili riparazioni delle maschere e analizzare i casi difficili.

L'ordine è quello di **un percorso di ricerca**, non di una singola applicazione di produzione: PCA, RANSAC, SAM, DeepLSD, re-ranking e gli altri esperimenti non vengono necessariamente eseguiti tutti nella stessa inferenza.

## 3. Perché abbiamo utilizzato RF-DETR?

Perché serviva studiare un modello che restituisse **istanze individuali**: non soltanto una classe «cavo», ma predizioni separate con maschere e punteggi.

RF-DETR è un approccio della famiglia DETR, con encoder visuale basato su Transformer e predizioni guidate da query. Nel notebook abbiamo sperimentato configurazioni diverse; una delle più articolate usa un encoder DINOv2 windowed-small, 1200 pixel di risoluzione, 200 query, 60 epoche e stabilizzazione tramite EMA e gradient accumulation.

**Motivazione della scelta:** distinguere istanze anche quando hanno la stessa categoria, sfruttando una rappresentazione visuale appresa.

**Trade-off:** la rappresentazione resta difficile quando i cavi sono sottilissimi, i dati sono densi e aumentare risoluzione, query o costo di training non garantisce automaticamente migliori risultati.

Non abbiamo inventato da zero l'architettura RF-DETR: **il contributo sperimentale del progetto è l'applicazione, la configurazione, la valutazione e l'esplorazione dei problemi e dei post-processing**, come documentato nel notebook.

## 4. Perché PCA e geometria sono importanti?

Immagina due maschere che descrivono approssimativamente lo stesso cavo, ma una è spostata lateralmente di due pixel. Se il cavo è spesso appena pochi pixel, l'IoU può scendere molto anche se entrambe seguono una direzione simile. Questo è un esempio concettuale, non una misura del dataset.

La PCA serve a individuare la **direzione di massima variazione dei pixel** che compongono una maschera. Da questa informazione il notebook costruisce rappresentazioni di linea, anche attraverso coordinate polari come `rho` e `theta`.

Questo permette di porre due domande differenti:

- **La maschera coincide con quella annotata?** — metriche COCO basate su matching e overlap.
- **La direzione/struttura lineare è coerente?** — misure geometriche derivate dalla maschera.

Ma attenzione: una direzione corretta **non significa** che il modello abbia separato correttamente tutte le istanze. Le due famiglie di metriche sono complementari.

## 5. Come abbiamo approfondito il problema

| Filone | Perché è stato esplorato | Cosa puoi raccontare |
| --- | --- | --- |
| **COCO dataset audit** | Capire le difficoltà prima di modificare il modello | Area delle maschere, rapporto bbox/mask, spessore, frammentazione, cavi per scena |
| **RF-DETR** | Avere predizioni a livello di singola istanza | Training, inferenza, scoring e metriche standard |
| **PCA** | Sfruttare la struttura prevalentemente lineare dei cavi | Direzione dominante, rappresentazione polare e ricostruzioni sperimentali |
| **RANSAC** | Rendere la stima della linea meno sensibile a punti anomali | Ipotesi di fitting robusto e scelte di soglia |
| **SAM** | Verificare possibili miglioramenti delle maschere partendo da predizioni esistenti | Esperimenti di mask refinement; alcuni hanno controlli oracle/GT |
| **DeepLSD / edge priors** | Esplorare la presenza di indizi geometrici nell'immagine | Confronto esplorativo con un rilevatore di strutture lineari |
| **Analisi per densità** | Non nascondere gli errori dietro una media globale | Scene semplici contro scene con molti cavi |

Il notebook conserva questi filoni come esperimenti separati. Non è corretto dire che tutti abbiano prodotto un incremento misurato delle prestazioni finali.

## 6. I risultati: come dirli correttamente

Dall'analisi COCO salvata nel notebook (cella di codice originale 78) compaiono, **per quella specifica valutazione storica**:

- **AP50 ≈ 0,528**
- **AR@10 ≈ 0,280** con massimo 10 rilevamenti per immagine.

È presente anche un valore **0,9989555** chiamato `angle_diff`, ma qui occorre essere rigorosi: **non è un errore angolare di 0,998°**. Il codice calcola una *similarità* usando la funzione `exp(-0.12 × differenza_angolare_in_radianti)` e poi ne fa una media. Uno score vicino a 1 indica una differenza contenuta secondo quella particolare definizione.

Un'analisi salvata nel notebook mostra inoltre un AP50 inferiore nelle immagini con molti cavi rispetto alle immagini meno dense. È una buona osservazione da discutere, **ma vale per lo specifico snapshot di predizioni analizzato**, non per ogni configurazione.

**Nota sulle metriche:** alcune celle esplorative scambiano gli indici di AR@1, AR@10 e AR@100 di COCOeval. Per parlare di performance numeriche aggiornate servirebbe un protocollo unico con etichette corrette.

## 7. Spiegazione da 30 secondi

> Insieme a un collega abbiamo studiato la segmentazione a istanze dei cavi elettrici nelle immagini aeree. È un problema impegnativo perché i cavi occupano pochissimi pixel, sono molto allungati e spesso si sovrappongono. Abbiamo sperimentato RF-DETR per riconoscere le istanze individuali e valutato i risultati con COCOeval. In seguito abbiamo studiato la geometria delle maschere, in particolare con PCA, e analizzato le difficoltà nelle immagini con molti cavi. L'aspetto interessante è che la qualità geometrica e le metriche di sovrapposizione raccontano aspetti diversi del problema.

## 8. Spiegazione tecnica da circa 90 secondi

> Il progetto nasce da un limite dei problemi di instance segmentation tradizionali quando gli oggetti sono filamentari. I cavi elettrici, ripresi dall'alto, hanno pochi pixel di spessore, possono essere parzialmente occlusi e diventano difficili da distinguere quando diversi cavi attraversano la stessa scena.
>
> Abbiamo utilizzato RF-DETR, un modello di instance segmentation basato su Transformer e query, per produrre maschere e score riferiti ai singoli cavi. Il notebook include analisi del dataset in formato COCO, configurazioni di training ad alta risoluzione, inferenza e valutazioni con COCOeval.
>
> Dopo la valutazione globale abbiamo analizzato i fallimenti: numero di cavi per immagine, frammentazione, spessore delle maschere, falsi positivi e falsi negativi. È emerso, negli output storici esaminati, che le scene più dense rappresentano un caso particolarmente difficile.
>
> Un secondo filone riguarda la geometria. Poiché i cavi hanno una direzione dominante, abbiamo sperimentato la PCA per ricavare una rappresentazione lineare dalla maschera, distinguendo la qualità della direzione dalla qualità dell'overlap. Il notebook conserva anche prove esplorative con RANSAC, SAM, DeepLSD e re-ranking delle predizioni.
>
> Il principale insegnamento è che, per oggetti sottili e allungati, non basta guardare una sola metrica globale. Bisogna distinguere la capacità del modello di identificare ogni istanza dalla qualità geometrica della ricostruzione. Per ottenere confronti quantitativi definitivi tra i post-processing servirebbe una valutazione ripetibile con dataset, checkpoint e protocolli uniformi.

## 9. Domande tecniche da colloquio

| Domanda | Risposta efficace |
| --- | --- |
| **Instance segmentation o semantic segmentation?** | Il problema richiede potenzialmente di distinguere due cavi della stessa classe come istanze diverse, non solo classificare i pixel come cavo/sfondo. |
| **Perché RF-DETR?** | È un modello query-based che produce predizioni a livello di istanza. Era un candidato interessante da studiare in scene con molte strutture simili. |
| **Perché lavorare ad alta risoluzione?** | I cavi sono pochi pixel di spessore: ridurre troppo l'immagine può far perdere dettaglio, a costo però di maggiore memoria e tempo. |
| **Perché IoU può essere fuorviante?** | Un piccolo spostamento laterale di una maschera sottilissima riduce sensibilmente i pixel condivisi anche se la direzione è simile. |
| **Che cosa restituisce la PCA?** | La direzione principale della distribuzione dei pixel della maschera; da lì si può ricavare una rappresentazione di linea. |
| **La PCA corregge sempre le maschere?** | No. È una stima geometrica e può semplificare eccessivamente cavi curvi, interrotti o sovrapposti. |
| **A cosa serve RANSAC?** | A provare un fitting di linea più robusto in presenza di punti anomali o segmentazioni rumorose. |
| **Perché SAM?** | Per verificare, in un ramo sperimentale, se un segmenter guidato da prompt potesse rifinire maschere prodotte da RF-DETR. |
| **Come hai misurato la qualità?** | COCO AP/AR per le maschere e analisi separate della geometria e del numero di cavi per scena. |
| **0,998 è un errore di circa 1 grado?** | No: è uno score di similarità angolare adimensionale definito nel notebook. Per dichiarare l'errore medio in gradi bisogna calcolarlo esplicitamente. |
| **Sono confronti equi tra tutte le varianti?** | Non ancora: alcune prove usano dataset, checkpoint o scelte di soglia differenti e alcune sono oracle diagnostici. |
| **Cosa faresti per migliorarlo?** | Congelerei split e checkpoint, sceglierei parametri su validation, rivaluterei le metriche corrette e confronterei singolarmente i post-processing per densità e spessore. |

## 10. Come presentare correttamente la collaborazione

Il progetto è attribuito nel README a **Paolo Pangallo e Gianluigi Oricchio**. Nei colloqui puoi quindi introdurlo come un progetto sviluppato insieme e, quando richiesto, specificare **quali parti hai implementato personalmente**. Il repository non contiene una suddivisione verificata dei contributi individuali: non sarebbe corretto inventarla.

## 11. Limiti da saper discutere

- Ci sono molte configurazioni storiche di training e dataset, non un unico run completamente riproducibile.
- Alcuni esperimenti fanno sweep di soglie su dati di test o si servono del ground truth per selezionare casi: sono studi esplorativi, non misure held-out di un algoritmo pronto all'uso.
- Le metriche di recall di alcune celle sono etichettate con indici COCO scorretti.
- Il valore di similarità angolare non può essere chiamato errore medio in gradi.
- Il checkpoint è gestito tramite Git LFS e il dataset non è distribuito direttamente nel repository.
- La documentazione non pretende di aver rieseguito i modelli, rigenerato le predizioni o dimostrato un miglioramento dei post-processing.

## 12. La frase che riassume il progetto

> **Abbiamo studiato la segmentazione di oggetti sottili e allungati, cercando di capire non soltanto quanto bene un modello predice le maschere, ma anche come la geometria e la densità delle scene influenzino la qualità delle istanze ricostruite.**
