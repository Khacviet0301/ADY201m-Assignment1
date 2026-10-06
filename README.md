# ADY201m Assignment – Cardiovascular Disease Analysis

Analysis of 70,000 patient records ([Kaggle dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)) to find factors related to cardiovascular disease.

## Files
| File | Content |
|---|---|
| `ADY201m-Assignment- Dataset-cardio_train_raw.csv` | Raw dataset (`;` separated) |
| `01_queries_sqlite.ipynb` | Load data into a SQL database and run the 10 queries of the assignment |
| `02_analysis.ipynb` | Cleaning, visualisation, statistical tests, OLS, classification models |
| `CARDIO_TRAIN.csv` | Table exported from the database |
| `images/` | Charts produced by the analysis notebook |
| `build_notebooks.py` | Script that generates the two notebooks |

## Note on IBM Cloud
IBM Cloud / Db2 / Watson Studio could not be used (account registration not possible), so
**SQLite (`sqlite3`) replaces IBM Db2 + `ibm_db`** and Jupyter/GitHub replace Watson Studio.
The SQL statements are the same as in the assignment. The original `ibm_db` connection code is kept as a comment in notebook 1.

## Main results
- Dataset is balanced: 34,979 of 70,000 people have the disease.
- Cleaning removed ~1,400 rows with impossible blood-pressure / height / weight values.
- Best model: **GradientBoosting, accuracy ≈ 73.5%** (Ridge baseline ≈ 72.5%).

## Run
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Run `01_queries_sqlite.ipynb` first (creates `CARDIO_TRAIN.csv`), then `02_analysis.ipynb`.
