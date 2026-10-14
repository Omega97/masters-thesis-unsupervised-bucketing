
# Cheat-sheet dei paper della tesi 📚

Ho recuperato il materiale della tua tesi Unsupervised State Bucketing for Mixture of Experts in Resource-Constrained Chess Engines e sto organizzando i riferimenti per idea, contributo scientifico e collegamento con il tuo lavoro.

La cosa importante da tenere a mente è che i paper non hanno tutti lo stesso ruolo: alcuni introducono le architetture che usi, altri motivano le tue scelte metodologiche, altri ancora sono lavori concorrenti con cui confrontare il tuo contributo.

## 1. Chess engines, NNUE e ricerca

Questi riferimenti costituiscono le fondamenta del problema: come valutare una posizione e come farlo in modo abbastanza efficiente da poterla usare durante la ricerca.

[Science Magazine - December 7, 2018 A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://images.openai.com/static-rsc-4/A5bAFgpxxrPKsi_QRmGMK1ANGVueJ7kWeluauy_y9sjDIhwMlvl1lYZjIB5Nkh8sShAKWS2FVR2JO1KxNWul23jF-ZakFIhg2Y2c94gdQOydFYCy9b6JkQTVmiQlnwUVPNKMg0CutxTjIh7y3OA9ZW3P6k7GVdU8OuOzWxe_nQQ?purpose=inline)

1\. Silver et al. (2018) — AlphaZero

[Paper originale](https://arxiv.org/abs/1712.01815)

Fondamenti

Contributo: dimostra che un sistema basato su deep learning e self-play può raggiungere prestazioni di livello mondiale in scacchi, shogi e Go, combinando rete neurale e Monte Carlo Tree Search (MCTS).

Collegamento con la tesi: ti serve per confrontare il paradigma AlphaZero, basato su MCTS, con gli engine classici basati su alpha-beta, e per contestualizzare la valutazione neurale nelle architetture di gioco. Non è una fonte specifica per il tuo algoritmo MoE.

[GitHub - HalfKP/NNUE: Efficiently Updatable Neural Network · GitHub](https://images.openai.com/static-rsc-4/vXgnZ7xTJJ0eM-8-VyX4IjF5YeNTQuoiVDtnNjsVIu07pKKsTQKxGjXf3x0-bvvPeHlwfuuRu3zVo-uuiqk28t8tqiG-bmA8jbM32Uu2smhpFugCDtj072JblbE5Cq0JlWHMWwL1LGwCv6DREzy32PPNBXjOdmu3d6o8w-RaTV4?purpose=inline)

2\. Nasu (2018) — NNUE

[Paper originale, traduzione inglese](https://github.com/asdfjkl/nnue)

Fondamenti

Contributo: introduce la valutazione neurale efficiently updatable per lo shogi. La caratteristica distintiva è la possibilità di aggiornare incrementalmente l'accumulatore sfruttando il fatto che una mossa modifica solo una parte limitata della posizione.

Collegamento con la tesi: è la base concettuale della tua architettura: input sparso, accumulatore L1 aggiornabile, head leggero e inferenza efficiente. La tua ricerca conserva questa struttura e studia come modificarne la specializzazione.

[Neurons Stockfish NNUE - Chess Stack Exchange](https://images.openai.com/static-rsc-4/5MLRnO6Gomnjh0UKO87hx4vIa7Sn8Hnpzsv9R0zdSEueK_tGHDQFwJTAYiuANeIpEw0smAzNS2B4GtAb9P_RFCXKb63d65Oexuc5vweUmqaaI5dC96Sn7eJM4CKF8YAuT261S2QPP2DWIXmBpiL5ZZ6C0OEQydAGqRORYC3FUFk?purpose=inline)

3\. Documentazione NNUE di Stockfish

[Documentazione ufficiale](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html) · [Repository NNUE-PyTorch](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)

Riferimento implementativo

Contributo: documenta la struttura pratica delle NNUE, la sparsità dell'input, l'aggiornamento incrementale e le scelte che rendono possibile l'inferenza a bassa latenza.

Collegamento con la tesi: giustifica le scelte architetturali e di quantizzazione. È una fonte tecnica, non un paper peer-reviewed. Per descrivere il codice concreto dell'engine, è utile anche il [repository ufficiale Stockfish](https://github.com/official-stockfish/Stockfish).

[GitHub - syzygy1/Cfish: C port of Stockfish · GitHub](https://images.openai.com/static-rsc-4/LO3jV9--jsVZewdCTYKC0vjvg-2Ix1d6FNzhfDN3YdveAAJPLYLgUa1RHbjr1aIv67Y2Eu1e55ChiE9CKr2krHsjHSxS57a-ACoX-drw9EMZAA9I8rPThwiWGR0ALiX1Mxpn2_6DN2YxfOC61lau-JquDfvMyhjYDbuoMLDKF9Y?purpose=inline)

4\. Cfish — de Man (2016)

[Repository](https://github.com/syzygy1/Cfish)

Implementazione

Contributo: porta il motore Stockfish in C, con attenzione alla portabilità e all'efficienza implementativa.

Collegamento con la tesi: è l'engine ospite nel quale integri il valutatore NNUE modificato. Ti permette di misurare non solo le metriche del modello, ma anche il costo della soluzione nel contesto effettivo della ricerca scacchistica.

Nota: questo è un riferimento software, non un paper scientifico.

[Leela Chess Zero Blog | Leela Chess Zero](https://images.openai.com/static-rsc-4/HwO0ldsmBTHzzHYtfDgdTpyfJymZEqhZ0WlDdG1SbCj4KxsCODmHiFlOflG4zMAgSusTyBMrqvTSMe3lXjPuO_yZWdfedYUrlgKoPIKXyQ3pAt-1cO9DtQja1jWNaD8G1CFmFNDfil77pYvadAUMceZiWy6LXWOrjDD2Cr5j1Bo?purpose=inline)

5\. Helfenstein et al. — M2CTS (2024 preprint; versione journal 2026)

[Preprint](https://arxiv.org/abs/2401.16852) · [Pubblicazione IEEE](https://www-live.dfki.de/en/web/research/projects-and-publications/publication/17385)

Confronto diretto

Contributo: studia una combinazione di Mixture of Experts e MCTS, con reti specializzate per differenti fasi della partita e diverse strategie di addestramento.

Collegamento con la tesi: mostra che la specializzazione MoE è già stata esplorata negli scacchi. La tua impostazione si distingue per l'uso di NNUE, per i vincoli embedded e per il confronto tra diversi criteri di bucketing, inclusi attivazioni e sample gradients.

6\. Ngo & Do (2026) — Hexaïssa

[Paper AAAI](https://doi.org/10.1609/aaai.v40i29.39636)

Contributo: esplora il routing tra diversi motori scacchistici mediante MoE e latent reward learning.

Collegamento con la tesi: è un esempio recente di selezione adattiva di esperti nel dominio scacchistico. Il confronto utile è tra selezionare motori diversi e selezionare head specializzate all'interno di un'architettura NNUE condivisa.

## 2. Mixture of Experts: routing e specializzazione

Questi lavori spiegano il principio architetturale alla base della tua MoE: più esperti specializzati, ma solo una parte della rete viene attivata per ciascun input.

[論文解説 Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (MoE) - ディープラーニングブログ](https://images.openai.com/static-rsc-4/OY__kBD0UXWISReB8w5wYaEA50vFJ4Y5gjVDMGT7-QZTo8AHzGeCGgUI8Nb-6S4lrfvTK68-KlT2_jvkk9va6QAqb9wUM15BzFXVjrw2xSthhrqhITja9OTifBg1kMQfL3DE4Xj9RuJh5xo_TlBSNgVBhonR6b1vzbD4ATWEh4k?purpose=inline)

7\. Shazeer et al. (2017) — Sparsely-Gated Mixture-of-Experts

[Paper](https://arxiv.org/abs/1701.06538)

Contributo: formalizza una forma moderna di MoE con numerosi esperti e gating sparso, consentendo di aumentare la capacità del modello senza attivare tutti gli esperti per ogni esempio.

Collegamento con la tesi: fornisce il principio di base della tua architettura. Il tuo vincolo è particolarmente severo: vuoi selezionare un singolo head economico, perché il valutatore viene invocato ripetutamente durante alpha-beta.

[Switch Transformers: To Trillion Parameter Models With Simple and Efficient | PDF](https://images.openai.com/static-rsc-4/s2oZMu5FyswHZLDD1Yu_iOllNZdn4TSJSbqjYTXJ_RY4eOgUm8nT-qgCJ_IO2D7y3uo_qjhCNjgE5qziaG3JlSHMCOIW8Irs2ctQZyOD3sey56YTB1PPkOzI47ahVG8Cn8X8Nif4rvnmZUtMeYYTeE1q4lFiSD4bwyIvy_ebj1A?purpose=inline)

8\. Fedus, Zoph & Shazeer (2021) — Switch Transformer

[Paper](https://arxiv.org/abs/2101.03961)

Contributo: semplifica il routing MoE usando principalmente l'assegnazione top-1: ciascun token viene indirizzato a un solo esperto.

Collegamento con la tesi: motiva la scelta di valutare un solo head per posizione, invece di combinare più reti. La distinzione centrale è tra memoria totale dei parametri e costo computazionale effettivo per input.

[GLaM: Efficient Scaling of Language Models with Mixture-of-Experts | DeepAI](https://images.openai.com/static-rsc-4/aSONjbX4sA_yKDBTkE_2kXcgIIMqPg1RUIP8wisy8dyLApSjHhPo0wuSZvWn1ADiAJ_aUP8uP0e7WPH1SL-o5mT8ebdPsiHQJ5wf7rxsf-DlHAnFaD1N1OgI4ENphdsKaicRNREbnn9FCVCV5aKVnIZ2oIiAhBhjDpXvH6eVFEU?purpose=inline)

9\. Du et al. (2022) — GLaM

[Paper PMLR](https://proceedings.mlr.press/v162/du22c.html)

Contributo: mostra come l'attivazione sparsa degli esperti possa supportare modelli linguistici di grandi dimensioni con un costo per input inferiore a quello che richiederebbe l'attivazione densa di tutti i parametri.

Collegamento con la tesi: rafforza la motivazione generale per la sparse activation. La differenza è che il tuo obiettivo non è scalare a miliardi di parametri, ma mantenere un valutatore molto piccolo e adatto a hardware con risorse limitate.

[Aman's AI Journal • Primers • Mixture of Experts](https://images.openai.com/static-rsc-4/HJHPB_s4yfSJAsJyWqpoRUiJfHLqlx0j73KEH249_u6Jq8no4kELAjihqHw0qrCAQlwx90sI2PPubY9bMGJl1l1keac3c1dYpHW_qU5n0YHGjZZ7wwi7wZQBy7Bvu5gMW847Hae2sHgQayDTSp6NHVEjt511PPfppCmP5u-kwls?purpose=inline)

10\. Kawata et al. (2025) — MoE e struttura latente dei cluster

[Paper ICML](https://proceedings.mlr.press/v267/kawata25a.html)

Contributo: studia teoricamente quando una MoE addestrata con stochastic gradient descent può rilevare e sfruttare una struttura latente a cluster, ottenendo vantaggi rispetto a un singolo modello in uno specifico setting.

Collegamento con la tesi: offre una motivazione teorica per la specializzazione: se il problema contiene sottostrutture differenti, esperti distinti possono impararle più efficacemente. Attenzione, però: i risultati teorici valgono sotto le ipotesi del paper e non dimostrano automaticamente che il tuo clustering di posizioni scacchistiche debba funzionare.

## 3. LoRA e gradient-based specialization

Questa è probabilmente la sezione più importante per spiegare dove si colloca scientificamente il tuo metodo. Questi lavori studiano come usare i gradienti per trovare gruppi di esempi che richiedono aggiornamenti simili, e come trasformare tale struttura in esperti specializzati.

[Mastering Low-Rank Adaptation (LoRA): Enhancing Large Language Models for Efficient Adaptation | DataCamp](https://images.openai.com/static-rsc-4/VhVCjPHSUNr3AHnD89J5id0WAy3kuZzojQodyZzRja0SPAENXP6kLRFH5HIQQBdGFaivL1XEH0PPQyQZnaA_jSP9Euq93EU-q47di5T9V9N5esgIUQUB_QCzZ8uKEfmdiH4z-I1cT74EN_IL-3A-8TJsi0-VMjnfR4bk3qpHpc0?purpose=inline)

11\. Hu et al. (2021) — LoRA

[Paper](https://arxiv.org/abs/2106.09685)

Contributo: introduce la Low-Rank Adaptation, che permette di adattare un modello congelato tramite aggiornamenti a basso rango, riducendo i parametri addestrabili.

Collegamento con la tesi: ti fornisce il contesto per capire come vengono rappresentati gli esperti nei lavori sui LLM. Tu non hai bisogno di LoRA, perché il tuo head NNUE è già piccolo: puoi duplicare direttamente i parametri di L2 e dell'output, condividendo L1.

[ICML Poster A Stronger Mixture of Low-Rank Experts for Fine-Tuning Foundation Models](https://images.openai.com/static-rsc-4/iFumVh-MKazFfTBJFlOj7zKSUbXjAw8ZGNViuXw3HMFbc4f4Nqv72JyCWrq9QJqOEpqsq-8xMsi4_nAT-0D4pJFMqlPQEu3YOtk8O7k2BeCw735nFBF2HQYS7IxlOg2awjPGC3ETWckODBNulK4JkhGlyP-22v1_7TUVABBH9TM?purpose=inline)

12\. Li et al. (2025) — ELREA

Ensembles of Low-Rank Expert Adapters

[Paper ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e8711daef520be07cb9852390c673de8-Abstract-Conference.html)

Precedente metodologico chiave

Contributo: raggruppa istruzioni di training in base alle direzioni dei gradienti, addestra esperti LoRA separati per cluster e usa una strategia di routing per selezionare gli esperti.

Collegamento con la tesi: è uno dei riferimenti più vicini alla tua idea di partizionare i dati secondo il segnale di apprendimento. La tua ricerca trasferisce il principio agli scacchi, usa gradienti rispetto al piccolo head NNUE e studia un dispatcher lineare compatibile con l'inferenza embedded.

Differenza importante: ELREA sfrutta informazioni di similarità dei gradienti nel routing. Nel tuo caso il gradiente della posizione non è disponibile durante l'inferenza perché richiede il target, quindi devi predire il bucket da informazioni forward-only.

[TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data - ACL Anthology](https://images.openai.com/static-rsc-4/pWewdiNu95UsFhDA6EfBHcZwT5PCpJfJe4arfPCHT9wwMKY6brVdx8jrTMEzw8V-ykeeEffGndGUiP_ysQekhJ52uTXhDO3O9VTnsTxBgJwyJV35YIMLfHx1R1i8GxP9VWEDINNi8Gf00GZbQlpPh3LXNxwIEMC8ohaMANQnJOk?purpose=inline)

13\. Sridharan et al. (2025) — GradientSpace

[Preprint](https://arxiv.org/abs/2512.06678)

Precedente metodologico chiave

Contributo: usa il clustering dei gradienti LoRA per identificare gruppi di istruzioni e addestra esperti specializzati; un router leggero consente di scegliere l'esperto usando le caratteristiche dell'input, senza dover calcolare il gradiente a runtime.

Collegamento con la tesi: è particolarmente rilevante per la sequenza gradient clustering → expert specialization → lightweight routing. La tua tesi verifica quanto di questa idea sia trasferibile a una rete NNUE molto più piccola e a un ambiente con vincoli di latenza e memoria più stringenti.

Da ricordare: la somiglianza dei gradienti non garantisce che la stessa partizione sia recuperabile con un semplice classificatore sulle attivazioni L1. Questa è una domanda sperimentale centrale nel tuo lavoro.

[LESS: Selecting Influential Data for Targeted Instruction Tuning | Papers with Code](https://images.openai.com/static-rsc-4/1tP0OxqrMBHu0hcoH-94QHHc8w92BBAEW09Qibu7MIKwrkamazJBlFYJfc23gtxOP5jxZK-GPKGs5cu4HfY9aDkFN6SRsAFrVWc_bHcXIt2X-K8Ykan_gzj0NxdTjtGMzbJQblCN7J-kczWowdslDGWYMYRBEkNc4tsBSmJUwEA?purpose=inline)

14\. Zhang et al. (2025) — TAGCOS

Task-Agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data

[Paper ACL Anthology](https://aclanthology.org/2025.findings-naacl.264/)

Contributo: rappresenta gli esempi attraverso i gradienti, li raggruppa e seleziona un sottoinsieme rappresentativo dei dati (coreset) per rendere più efficiente il fine-tuning.

Collegamento con la tesi: supporta l'idea che i gradienti possano essere utilizzati come rappresentazione dei dati, non soltanto come strumento per aggiornare i pesi. Tu usi lo stesso principio per costruire bucket di posizioni, ma il tuo obiettivo è la specializzazione degli esperti, non la selezione di un sottoinsieme del dataset.

Nota bibliografica: la versione pubblicata è nei Findings of NAACL 2025, quindi aggiorna l'anno da 2024 a 2025 nella bibliografia, se stai citando questa versione.

[Abir Harrasse](https://images.openai.com/static-rsc-4/Es7yn3xpFXzeF7WRjYG4JwHD2LhWmSkcrtJ7bjM43h6u1H_rhhXiEkzuu4hdd3f8LxQem33TmlftbDblCn_BrxDMEKRlJlcjnN4dlEdARaZB5aub8yVNmKgk0o5Nwyye_pacVAk4HKqvwaPXxQ6gRGD5Pi0cnpTcUOFcW3lcOME?purpose=inline)

15\. Rosser (2026) — Gradient Atoms

[Repository e materiale del progetto](https://github.com/jrosseruk/gradient_atoms)

Contributo: esplora la decomposizione dei gradienti in componenti sparse per scoprire e attribuire comportamenti dei modelli.

Collegamento con la tesi: è un riferimento complementare per l'idea che i gradienti contengano struttura informativa interpretabile. Non è però un precedente diretto per la tua MoE: il suo obiettivo principale è la scoperta e l'attribuzione dei comportamenti, non il routing di esperti NNUE.

Attenzione: se lo citi come paper scientifico, usa anche la relativa pubblicazione o il preprint, non soltanto il repository.

## 4. Clustering, valutazione e fondamenti del deep learning

Questi lavori supportano i metodi di analisi che usi per capire se i bucket siano effettivamente strutturati e se la partizione sia utile.

[(PDF) Rousseeuw, P.J.: Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Comput. Appl. Math. 20, 53-65](https://images.openai.com/static-rsc-4/5X9FjZcHFErbhsuKCMJxHwZwbMsHRMZgA1pa11f0NgiergsIdaZaZQuMkn5zbgCvNuVZQbUwHMQRv6hjtIayjcNlGwPQMp7PYfK8wGABOLlvg2UQaDwWifwm7rtPy0EM0ZUjlIBlwF8Zo765WMAmImcSa81qrWklkDQW-4krRlU?purpose=inline)

16\. Rousseeuw (1987) — Silhouette score

[Articolo](https://doi.org/10.1016/0377-0427\(87\)90125-7)

Contributo: introduce la silhouette come misura della coesione di un cluster rispetto alla separazione dagli altri cluster.

Collegamento con la tesi: ti permette di confrontare quantitativamente le partizioni ottenute da board encoding, attivazioni L1 e sample gradients. Una silhouette più alta indica una separazione geometrica migliore, ma non dimostra da sola che gli esperti risultanti avranno prestazioni superiori.

[2018-Clustering by Fast Search and Find of Density Peaks | PDF | Cluster Analysis | Probability Distribution](https://images.openai.com/static-rsc-4/fWtHkhI23DALJtvrnYVqchM_g3fIPVnTPOjIL7mXAbns0asz4fxWZqWu-746nzGpvAbc0Pq30Fv2NMechqXpETiset06jgpTVd2YDTWc6XNrfmVByX8TtE1Tp7dloBogrYbz6nMvW34YaaOFYZ_swG7J500pVOKjOKbyYnsxwlU?purpose=inline)

17\. Rodriguez & Laio (2014) — Density Peak Clustering

[Articolo su Science](https://doi.org/10.1126/science.1242072)

Contributo: introduce un metodo di clustering basato sulla densità locale e sulla distanza da punti a densità superiore, pensato per individuare cluster senza imporre una forma geometrica semplice.

Collegamento con la tesi: è pertinente alla scelta di algoritmi alternativi a K-Means, in particolare quando si vuole esplorare se i gradienti formino gruppi non sferici o strutture più complesse. Se il metodo viene solo discusso ma non utilizzato negli esperimenti finali, presentalo come alternativa metodologica, non come componente del sistema finale.

[Deep Learning Book By Ian Goodfellow,yoshua Bengio,aaron Courville, ('tc') | Indigo](https://images.openai.com/static-rsc-4/--xhr34f9Kyae9BJ6FwjEO_dD9r6BBAI2_PE28McyRzTAel1N1ZG2aEnIRfQdnBvZG3_uMcz8zWF3aBhNTlnG9R20u2cbcgBFSBIQQt9hdckwy-4i6_wqINgBUc5nsMVa7QXvA4zM2MZ57KOBKn0RilsJVbDrjg1-Xl3XAsF9BA?purpose=inline)

18\. Goodfellow, Bengio & Courville (2016) — Deep Learning

[Libro online](https://www.deeplearningbook.org/) · [MIT Press](https://mitpress.mit.edu/9780262337373/deep-learning/)

Contributo: testo di riferimento per reti neurali, backpropagation, ottimizzazione, funzioni di perdita e generalizzazione.

Collegamento con la tesi: è una fonte teorica generale per le spiegazioni del training e dei gradienti. Non è una fonte specifica per NNUE o per il tuo metodo di clustering.

[BURKOV (@burkov) on X](https://images.openai.com/static-rsc-4/I0BgUb44Yh9WWxGy300PDwvXHcYO5OKbkgiVRVDAeK_LgrdgYrQmy0bCOro1tZY4GWSv3NzJNBXsV6oNF1gTwKIRryjn_WupyPIDdsBlSduFSgJ9dDROB0JwqliE8E_EwHTrUAHxX90eZBhRe-Rh9ikOWr0gykp4C1M-cs5kgP0?purpose=inline)

19\. Hinton, Vinyals & Dean (2015) — Knowledge Distillation

[Paper](https://arxiv.org/abs/1503.02531)

Contributo: descrive come trasferire informazioni da un modello insegnante (teacher) a un modello studente, usando le uscite del teacher come supervisione.

Collegamento con la tesi: è pertinente quando descrivi l'addestramento dell'NNUE con distribuzioni WDL fornite da un engine teacher. Attenzione alla terminologia: usare soft target di un teacher è compatibile con la distillazione, ma non implica che tu stia replicando l'intero algoritmo originale di Hinton.

## 5. Riferimenti per il mondo scacchistico e l'implementazione

Questi non sono tutti paper scientifici, ma possono essere citazioni necessarie per documentare dati, software e scelte di implementazione.

| Riferimento                                                                                                                        | Perché compare nella tesi                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Leela Chess Zero (Lc0)](https://lczero.org/)                                                                                      | Engine neurale usato come fonte di valutazioni/target WDL, secondo la configurazione sperimentale descritta nella tesi.                                                     |
| [Kaggle FIDE & Google Efficient Chess AI Challenge](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge) | Contesto applicativo di motori scacchistici efficienti e valutatori adatti a vincoli computazionali. Da citare solo se la descrizione del challenge resta nel testo finale. |
| [YaneuraOu](https://github.com/yaneurao/YaneuraOu)                                                                                 | Riferimento software per l'origine e lo sviluppo di NNUE nello shogi.                                                                                                       |
| [Stockfish NNUE networks](https://github.com/official-stockfish/networks)                                                          | Fonte dei network distribuiti, se descrivi o confronti pesi specifici.                                                                                                      |
| [PeSTO evaluation function](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function)                                        | Esempio di valutazione scacchistica classica, utile se discuti funzioni di valutazione manuali. È una pagina tecnica, non un paper peer-reviewed.                           |

## 6. World models: un punto da completare nella bibliografia

La tua tesi considera anche la possibilità di sostituire l'accumulatore NNUE convenzionale con un learned world model, addestrato senza feedback diretto del teacher engine o dei risultati delle partite.

Nella bibliografia che ho recuperato non vedo però una citazione specifica e completa per questa parte. Non voglio attribuirle un paper che potrebbe non essere quello effettivamente usato.

Per dare un fondamento bibliografico a questa sezione, verifica quale lavoro hai seguito per il world model. Un riferimento possibile, se la tua implementazione è concettualmente vicina ai modelli di dinamica latente usati nel reinforcement learning, è [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)](https://arxiv.org/abs/1911.08265). Ma MuZero non è automaticamente la fonte corretta per il tuo metodo: va citato solo se la relazione teorica è effettiva.

## 7. Le 8 reference da conoscere meglio per la discussione

Se devi ripassare rapidamente per la discussione di laurea, concentrati su queste:

Nasu / NNUE

Devi saper spiegare perché l'accumulatore si aggiorna incrementalmente e perché questo è importante per l'inferenza a basso costo.

Shazeer et al. / MoE

Devi saper spiegare perché avere molti esperti non significa doverli valutare tutti per ogni posizione.

Fedus et al. / Switch Transformer

Devi saper spiegare il routing top-1 e il compromesso tra capacità totale, memoria e computazione per input.

Li et al. / ELREA

Devi saper spiegare come i gradienti vengono usati per trovare gruppi di dati con esigenze di apprendimento simili e perché questo è rilevante per la tua ipotesi.

Sridharan et al. / GradientSpace

Devi saper spiegare il legame tra clustering dei gradienti, esperti specializzati e router leggero, distinguendo il loro setting dal tuo.

Kawata et al. / struttura latente

Devi saper spiegare cosa dimostra teoricamente il lavoro e, altrettanto importante, perché i risultati non garantiscono il successo del tuo metodo.

Rousseeuw / silhouette

Devi saper interpretare il punteggio e spiegare perché una buona separazione geometrica non equivale necessariamente a una migliore valutazione scacchistica.

Helfenstein et al. / M2CTS

Devi saper confrontare la specializzazione per fase della partita con la tua specializzazione data-driven di head NNUE, tenendo conto delle differenze fra MCTS e alpha-beta.

### Tre precisazioni bibliografiche importanti

- TAGCOS: la versione pubblicata che ho trovato è del 2025, non del 2024.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://aclanthology.org\&sz=32)

  ACL Anthology

- M2CTS: la versione del 2024 è un preprint; è stata pubblicata una versione journal nel 2026. Se citi la versione definitiva, aggiorna autori, anno e dati bibliografici.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://www-live.dfki.de\&sz=32)

  www-live.dfki.de

- GradientSpace: il link presente nella tua bibliografia punta a un preprint identificato come `arXiv:2512.06678`. Verifica che la data della versione che hai effettivamente consultato sia coerente con l'anno riportato nel testo.



Ecco la reference originale di RPROP in formato autore–anno, coerente con lo stile bibliografico della tua tesi. Il paper è del 1993, pubblicato negli atti dell'IEEE International Conference on Neural Networks, pp. 586–591.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://dblp.org\&sz=32)

dblp.org

+1



Riedmiller, M., & Braun, H. (1993). A direct adaptive method for faster backpropagation learning: The RPROP algorithm. *Proceedings of the IEEE International Conference on Neural Networks*, 586–591. [https://doi.org/10.1109/ICNN.1993.298623](https://doi.org/10.1109/ICNN.1993.298623)

Citazione nel testo: (Riedmiller & Braun, 1993).

Se usi BibTeX:

```
@inproceedings{riedmiller1993rprop,
  author    = {Riedmiller, Martin and Braun, Heinrich},
  title     = {A Direct Adaptive Method for Faster Backpropagation Learning: The {RPROP} Algorithm},
  booktitle = {Proceedings of the IEEE International Conference on Neural Networks},
  pages     = {586--591},
  year      = {1993},
  doi       = {10.1109/ICNN.1993.298623}
}
```

Puoi consultare il [PDF del paper originale](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Rprop.pdf).


---

## Notes

- **SIMD** sta per ==**Single Instruction, Multiple Data** (Singola Istruzione, Dati Multipli)==. Si tratta di una tecnologia di calcolo parallelo utilizzata dai moderni processori (CPU e GPU) per elaborare **più dati contemporaneamente con un'unica istruzione**.
    
- Le estensioni **AVX2** e **AVX-512** sono set di istruzioni SIMD di Intel e AMD. Consentono di elaborare stringhe di bit molto grandi (256 o 512 bit) in un colpo solo. Questo permette al software di calcolare le mosse possibili sulla scacchiera o valutare la sicurezza dei pezzi in parallelo, abbattendo drasticamente i tempi di calcolo.