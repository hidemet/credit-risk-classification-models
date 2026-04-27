# CreditRisk Sentinel

## Title & Elevator Pitch
CreditRisk Sentinel is a practical proof-of-concept for credit default classification on imbalanced tabular data.
It benchmarks multiple ML families, applies class-imbalance strategies, and optimizes for high-risk applicant detection quality (the `bad` class) instead of relying on accuracy alone.

## Project Positioning
This repository is built as an end-to-end risk modeling benchmark: from data profiling to model selection, hyperparameter optimization, and comparative evaluation.

## Tech Stack
- Python (Jupyter Notebook workflow)
- Pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE, ADASYN, RandomOver, RandomUnder, SMOTETomek)
- XGBoost
- SciPy, statsmodels
- Matplotlib, Seaborn
- mlxtend (Sequential Feature Selector)

## Architecture & Model Workflow
The project follows a clear ML execution flow:

1. Data ingestion and semantic mapping of German Credit categorical codes.
2. Data validation (missing values, duplicates, unexpected values).
3. Stratified train/test split to preserve class ratio.
4. EDA + statistical checks (ANOVA for numeric variables, chi-square for categorical variables).
5. Preprocessing:
     - log transforms for skewed numeric features,
     - scaling,
     - ordinal + one-hot encoding via transformers.
6. Baseline benchmark across seven classifiers.
7. Model-specific hyperparameter optimization with CV focused on `F1 (bad)`.
8. Imbalance handling experiments (resampling and weighting strategies).
9. Final test-set comparison and ROC-based ranking.
10. Optional 4-feature reduction with SFS for compact model variants.

## Dataset
- Primary source used by the notebook: German Credit dataset (`german.data`).
- Notebook default: loads from UCI URL.
- Local copies are available under `dataset/`.

Target classes:
- `good`: lower risk
- `bad`: higher risk

Primary positive class for decision quality in this project: `bad`.

## Key Achievements & Challenges

### Verified Baseline Results (Test Set)

| Model | Test Accuracy | Test F1 (bad) | Precision (bad) | Recall (bad) | CV F1 (bad) |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.5500 | 0.2500 | 0.2500 | 0.2500 | 0.3042 |
| Logistic Regression | 0.7100 | 0.4630 | 0.5208 | 0.4167 | 0.4306 |
| Decision Tree | 0.6950 | 0.4959 | 0.4918 | 0.5000 | 0.4739 |
| Random Forest | 0.7250 | 0.4086 | 0.5758 | 0.3167 | 0.4722 |
| SVM (RBF) | 0.7200 | 0.3913 | 0.5625 | 0.3000 | 0.4210 |
| AdaBoost | 0.6800 | 0.4182 | 0.4600 | 0.3833 | 0.4709 |
| XGBoost | 0.6950 | 0.4786 | 0.4912 | 0.4667 | 0.5425 |

### Best Tuned Configurations (from notebook outputs)

| Family | Best Method | Best CV F1 (bad) | Test Accuracy | Test F1 (bad) | Precision (bad) | Recall (bad) |
|---|---|---:|---:|---:|---:|---:|
| Decision Tree | `PrePruning_SMOTE` | 0.5961 | 0.6200 | 0.5250 | 0.4200 | 0.7000 |
| Random Forest | `RF_RandomUnder` | 0.6120 | 0.7100 | 0.6329 | 0.5102 | 0.8333 |
| SVM | `SVC_None` | 0.5878 | N/A in final SVM block | N/A in final SVM block | N/A in final SVM block | N/A in final SVM block |
| Logistic Regression | `LR_RandomUnder` | 0.5921 | 0.6600 | 0.5802 | 0.4608 | 0.7833 |
| AdaBoost | `Ada_SMOTETomek` | 0.6327 | 0.7000 | 0.5588 | 0.5000 | 0.6333 |
| XGBoost | `XGB_ScalePosWeight` | 0.6169 | 0.6700 | 0.5600 | 0.4667 | 0.7000 |

### Final Outcome
- Strongest operating point for high-risk detection was achieved by tuned Random Forest (`RF_RandomUnder`):
    - `F1 (bad) = 0.6329`
    - `Recall (bad) = 0.8333`
- Relative lift vs best baseline F1(bad) (Decision Tree 0.4959): about **+27.6%**.
- ROC ranking (AUC, full-feature models):
    - Random Forest: 0.7789
    - XGBoost: 0.7569
    - Logistic Regression: 0.7495

### Main Technical Challenges Solved
- Class imbalance (`good`/`bad` = 70/30) made plain accuracy misleading.
- Precision/recall trade-off on `bad` required metric-driven optimization around F1(recall-sensitive) and resampling.
- Overfitting pressure on tree-based models was handled through constrained search spaces and validation loops.

## Local Setup & Installation

### 1) Clone and enter the repository
```bash
git clone <your-repo-url>
cd credit-risk-classification-models
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
```

- Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

- Linux/macOS:
```bash
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy statsmodels xgboost mlxtend jupyter
```

### 4) Launch notebook
```bash
jupyter notebook credit_risk_analysis.ipynb
```

### 5) Data source option
The notebook currently loads the dataset from UCI URL.
If you prefer local data, set the `url` variable to `./dataset/german.data` in the notebook.

## UI / Demo

Add your visuals here:

```markdown
![Baseline confusion matrices](./assets/demo/baseline-confusions.png)
![Model comparison chart](./assets/demo/model-comparison.png)
![Best tuned model confusion matrix](./assets/demo/rf-randomunder-confusion.png)
![ROC curves](./assets/demo/roc-comparison.png)
![Feature importance](./assets/demo/rf-feature-importance.png)
```

Suggested order for recruiters:
1. Model comparison chart
2. Best tuned confusion matrix
3. ROC curve comparison
4. Feature importance plot

## Future Work / Roadmap

1. Refactor notebook logic into a reusable `src/` package with modular pipelines.
2. Add automated tests (currently missing) for preprocessing, data validation, and metric computations.
3. Add experiment tracking (e.g., MLflow/W&B) for reproducibility and candidate selection governance.
4. Introduce probability-threshold optimization for cost-sensitive decisioning.
5. Add calibration and monitoring hooks (drift checks, class-prior shifts).
6. Expose the best model as a lightweight inference API.

## License
See `LICENSE` for usage terms.
