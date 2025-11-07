# Modelli di Classificazione per il Rischio di Credito (Credit Risk)

Questo repository contiene un'analisi comparativa di diversi modelli di Machine Learning per la classificazione del rischio creditizio. Partendo dal dataset "German Credit", il progetto segue un intero pipeline di Data Science, dall'analisi esplorativa dei dati (EDA) fino alla valutazione delle performance dei modelli, per determinare il classificatore più efficace nel predire se un richiedente di prestito rappresenti un rischio "Buono" o "Cattivo".

➡️ **[Esplora il Notebook Completo (`.ipynb`) con tutto il codice e l'analisi](./credit_risk_analysis.ipynb)** ⬅️

## 🎯 Obiettivo di Business

L'obiettivo è sviluppare un modello predittivo per supportare le decisioni di concessione di prestiti. Classificando accuratamente i richiedenti in categorie a basso rischio (Good) e ad alto rischio (Bad), un istituto finanziario può **minimizzare le perdite dovute a insolvenze** e ottimizzare il proprio portafoglio crediti.

## 🔬 Processo di Analisi e Modellazione

Il progetto è strutturato seguendo le fasi standard di un ciclo di vita di Data Science:

1.  **Analisi Esplorativa dei Dati (EDA):** Investigazione iniziale del dataset per comprendere la distribuzione delle feature, le correlazioni e le caratteristiche principali dei richiedenti a "basso" e "alto" rischio.
2.  **Data Cleaning e Preprocessing:**
    *   Gestione dei valori mancanti e codifica delle variabili categoriche tramite `LabelEncoder`.
    *   Standardizzazione delle feature numeriche utilizzando `StandardScaler` per garantire che i modelli non siano influenzati dalla diversa scala dei dati.
3.  **Confronto tra Modelli di Classificazione:** Sono stati implementati, addestrati e valutati cinque diversi algoritmi di apprendimento supervisionato per confrontarne le performance su questo specifico problema:
    *   **Regressione Logistica:** Un modello lineare robusto e interpretabile.
    *   **K-Nearest Neighbors (KNN):** Un modello non parametrico basato sulla vicinanza dei dati.
    *   **Support Vector Machine (SVM):** Sia con kernel lineare che non lineare (RBF).
    *   **Naive Bayes:** Un classificatore probabilistico basato sul teorema di Bayes.
4.  **Valutazione delle Performance:** Ogni modello è stato valutato utilizzando l'**accuracy score** e l'analisi della **matrice di confusione** per comprendere non solo l'accuratezza generale, ma anche la tipologia di errori commessi (es. falsi positivi vs. falsi negativi).

## 🏆 Risultati e Conclusioni

L'analisi comparativa ha evidenziato che, sebbene diversi modelli abbiano raggiunto performance simili, il modello **[Inserire qui il modello migliore, es. Kernel SVM o Logistic Regression]** ha mostrato il miglior equilibrio tra accuratezza e interpretabilità per questo specifico dataset. I risultati dettagliati, le matrici di confusione e le conclusioni sono disponibili nel notebook.

Questo studio dimostra la capacità di applicare un processo metodico per risolvere un problema di business reale, selezionando e validando l'approccio di Machine Learning più appropriato.

## 🛠️ Stack Tecnologico

*   **Linguaggio:** Python
*   **Librerie Principali:**
    *   **Pandas:** Per la manipolazione e l'analisi dei dati.
    *   **NumPy:** Per le operazioni numeriche.
    *   **Scikit-learn:** Per l'implementazione dei modelli di ML, il preprocessing e la valutazione.
    *   **Matplotlib & Seaborn:** Per la visualizzazione dei dati e dei risultati.
