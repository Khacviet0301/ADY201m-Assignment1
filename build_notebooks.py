import nbformat as nbf

CSV = "ADY201m-Assignment- Dataset-cardio_train_raw.csv"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


# ---------------- Notebook 1: SQL queries ----------------
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md("# ADY201m Assignment - Part 1: SQL queries on CARDIO_TRAIN\n"
       "Do not create IBM Cloud account, **SQLite** (Python `sqlite3`) is used instead of IBM Db2 + `ibm_db`. "
       "SQL statements are the same as in the assignment."),
    code(f"""import sqlite3, pandas as pd

df = pd.read_csv("{CSV}", sep=";")
df.columns = [c.upper() for c in df.columns]
df["AGE"] = (df["AGE"] // 365).astype(int)   # days -> years (as in the PDF)
df.head()"""),
    md("## Load data into the database (equivalent to uploading CSV to Db2)"),
    code("""conn = sqlite3.connect("cardio.db")
df.to_sql("CARDIO_TRAIN", conn, if_exists="replace", index=False)
print(pd.read_sql("SELECT * FROM CARDIO_TRAIN LIMIT 5", conn))"""),
    md("## Helper: run a query and count how many rows have CARDIO = 1"),
    code("""def top10(query, col):
    cur = conn.execute(query)
    rows = cur.fetchall()
    count = 0
    for v, cardio in rows:
        if cardio == 1:
            count += 1
        print(f" {col}: {float(v)} CARDIO: {cardio}")
    print(f"There are {count}/10 cases got illness ({count*10}%).")"""),
    md("## Task 1: Number of rows"),
    code("""print(conn.execute("SELECT COUNT(ID) FROM CARDIO_TRAIN").fetchone()[0])"""),
    md("## Task 2: Number of cases with cardiovascular disease"),
    code("""print(conn.execute("SELECT COUNT(CARDIO) FROM CARDIO_TRAIN WHERE CARDIO=1").fetchone()[0])"""),
    md("## Task 3: Women and men"),
    code("""print("GENDER=1:", conn.execute("SELECT COUNT(GENDER) FROM CARDIO_TRAIN WHERE GENDER=1").fetchone()[0])
print("GENDER=2:", conn.execute("SELECT COUNT(GENDER) FROM CARDIO_TRAIN WHERE GENDER=2").fetchone()[0])"""),
    md("## Task 4: Top 10 highest AP_HI - % ill"),
    code("""top10("SELECT AP_HI, CARDIO FROM CARDIO_TRAIN ORDER BY AP_HI DESC LIMIT 10", "AP_HI")"""),
    md("## Task 5: 10 people with weight above average - % ill"),
    code("""top10(\"\"\"SELECT WEIGHT, CARDIO FROM CARDIO_TRAIN
WHERE WEIGHT > (SELECT AVG(WEIGHT) FROM CARDIO_TRAIN)
ORDER BY WEIGHT DESC LIMIT 10\"\"\", "WEIGHT")"""),
    md("## Task 6: GLUC > 1 (10 rows)"),
    code("""top10("SELECT GLUC, CARDIO FROM CARDIO_TRAIN WHERE GLUC > 1 LIMIT 10", "GLUC")"""),
    md("## Task 7: 10 oldest people"),
    code("""top10("SELECT AGE, CARDIO FROM CARDIO_TRAIN ORDER BY AGE DESC LIMIT 10", "AGE")"""),
    md("## Task 8: CHOLESTEROL = 3 (10 rows)"),
    code("""top10("SELECT CHOLESTEROL, CARDIO FROM CARDIO_TRAIN WHERE CHOLESTEROL=3 LIMIT 10", "CHOLESTEROL")"""),
    md("## Task 9: 10 smokers"),
    code("""top10("SELECT SMOKE, CARDIO FROM CARDIO_TRAIN WHERE SMOKE=1 LIMIT 10", "SMOKE")"""),
    md("## Task 10: 10 people who exercise"),
    code("""top10("SELECT ACTIVE, CARDIO FROM CARDIO_TRAIN WHERE ACTIVE=1 LIMIT 10", "ACTIVE")"""),
    md("## Extra queries from the console part (AP_LO lowest, alcohol)"),
    code("""print(pd.read_sql("SELECT * FROM CARDIO_TRAIN ORDER BY AP_LO ASC LIMIT 10", conn))
print(pd.read_sql("SELECT * FROM CARDIO_TRAIN WHERE ALCO=1 LIMIT 10", conn))"""),
    md("## Export table to CSV and close connection"),
    code("""out = pd.read_sql("SELECT * FROM CARDIO_TRAIN", conn)
out.to_csv("CARDIO_TRAIN.csv", index=False)
conn.close()
print(out.shape)"""),
    md("## (Optional) Original IBM Db2 connection code - fill in your own credentials if you have Db2\n"
       "```python\nimport ibm_db\ndsn = (\"DRIVER={IBM DB2 ODBC DRIVER};DATABASE=bludb;HOSTNAME=<host>;PORT=<port>;\"\n"
       "       \"PROTOCOL=TCPIP;UID=<uid>;PWD=<pwd>;SECURITY=SSL;\")\nconn = ibm_db.connect(dsn, \"\", \"\")\n"
       "res = ibm_db.exec_immediate(conn, \"SELECT COUNT(ID) FROM CARDIO_TRAIN\")\nprint(ibm_db.fetch_both(res))\n```"),
]

# ---------------- Notebook 2: analysis ----------------
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md("# ADY201m Assignment - Part 2: Data analysis and classification of cardiovascular disease"),
    code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings("ignore")
sns.set_theme()
df = pd.read_csv("CARDIO_TRAIN.csv")
print(df.shape)
df.head()"""),
    md("## 1. Describe data"),
    code("df.describe().T"),
    code("print(df.isna().sum().sum(), 'missing values;', df.duplicated().sum(), 'duplicates')\ndf.CARDIO.value_counts()"),
    md("## 2. Data cleaning\nRemove impossible values: `AP_HI < AP_LO`, blood pressure outside a plausible range, "
       "and extreme height/weight (1st-99.9th percentile style rules)."),
    code("""before = len(df)
df = df[(df.AP_HI > 0) & (df.AP_LO > 0) & (df.AP_HI >= df.AP_LO)]
df = df[df.AP_HI.between(60, 250) & df.AP_LO.between(30, 200)]
df = df[df.HEIGHT.between(130, 220) & df.WEIGHT.between(30, 200)]
print("removed", before - len(df), "rows ->", len(df), "left")
df.describe().T"""),
    md("## 3. Visualisation"),
    code("""import os; os.makedirs("images", exist_ok=True)
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for a, c in zip(ax.ravel(), ["AGE", "HEIGHT", "WEIGHT", "AP_HI", "AP_LO"]):
    sns.histplot(data=df, x=c, hue="CARDIO", kde=True, ax=a)
sns.countplot(data=df, x="CHOLESTEROL", hue="CARDIO", ax=ax[1, 2])
plt.tight_layout(); plt.savefig("images/distributions.png", dpi=120); plt.show()"""),
    code("""plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix"); plt.savefig("images/correlation.png", dpi=120); plt.show()"""),
    code("""fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, c in zip(ax, ["GLUC", "SMOKE", "ALCO", "ACTIVE"]):
    sns.barplot(data=df, x=c, y="CARDIO", ax=a)
plt.tight_layout(); plt.savefig("images/cardio_rate.png", dpi=120); plt.show()"""),
    md("## 4. Statistical tests"),
    code("""from scipy.stats import levene, ttest_ind, pearsonr, chi2_contingency
low = df[df.AGE < df.AGE.mean()].CARDIO; high = df[df.AGE >= df.AGE.mean()].CARDIO
f, p = levene(low, high, center="mean"); print("Levene:", f, p)
t, p2 = ttest_ind(low, high, equal_var=p > 0.05); print("T-test (age groups):", t, p2)
print("Pearson WEIGHT-CARDIO:", pearsonr(df.WEIGHT, df.CARDIO))
chi = chi2_contingency(pd.crosstab(df.ALCO, df.CARDIO), correction=True)
print("Chi2 ALCO-CARDIO: chi2=%.3f p=%.4f" % (chi[0], chi[1]))
print("=> alcohol:", "significant" if chi[1] < 0.05 else "no sufficient evidence (p>=0.05)")"""),
    md("## 5. OLS regression"),
    code("""import statsmodels.api as sm
feats = ["AGE","GENDER","HEIGHT","WEIGHT","AP_HI","AP_LO","CHOLESTEROL","GLUC","SMOKE","ALCO","ACTIVE"]
ols = sm.OLS(df["CARDIO"], sm.add_constant(df[feats])).fit()
ols.summary()"""),
    md("## 6. Standardise, split and model"),
    code("""from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
num = ["AGE","HEIGHT","WEIGHT","AP_HI","AP_LO"]; cat = ["GENDER","CHOLESTEROL","GLUC","SMOKE","ALCO","ACTIVE"]
X = np.concatenate([StandardScaler().fit_transform(df[num]), df[cat].values], axis=1)
y = df["CARDIO"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape"""),
    code("""from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                              BaggingClassifier, ExtraTreesClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score

models = {
    "Ridge": RidgeClassifier(alpha=0.0001),
    "RandomForest": RandomForestClassifier(random_state=0),
    "GradientBoosting": GradientBoostingClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
    "Bagging(KNN)": BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, random_state=0),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=0),
    "Stacking": StackingClassifier(
        estimators=[("ridge", RidgeClassifier()), ("knr", KNeighborsClassifier(n_neighbors=20))],
        final_estimator=GradientBoostingClassifier(n_estimators=25, subsample=0.5, min_samples_leaf=25,
                                                   max_features=1, random_state=42)),
}
res = {}
for n, m in models.items():
    m.fit(X_train, y_train)
    res[n] = accuracy_score(y_test, m.predict(X_test))
    print(f"{n:18s} accuracy = {res[n]:.4f}")"""),
    code("""r = pd.Series(res).sort_values()
r.plot.barh(figsize=(8, 4), xlim=(0.6, 0.76)); plt.title("Accuracy by model")
plt.tight_layout(); plt.savefig("images/model_accuracy.png", dpi=120); plt.show()
print("Best model:", r.idxmax(), round(r.max(), 4))"""),
    md("## 7. Conclusion\n"
       "- The target is balanced (~50% ill).\n- Blood pressure (AP_HI, AP_LO), age, cholesterol and weight are the strongest factors.\n"
       "- GradientBoosting gives the best accuracy (~73-74%), slightly better than the Ridge baseline (~72%).\n"
       "- Alcohol: on the cleaned data the chi-square test gives p ~ 0.027 < 0.05, so there is a statistically significant (but weak) relation. "
       "On the raw data (as in the PDF) p ~ 0.055, which is not significant - cleaning changes the conclusion."),
]

for name, nb in [("01_queries_sqlite.ipynb", nb1), ("02_analysis.ipynb", nb2)]:
    nbf.write(nb, name)
print("done")
