EXPERIMENT 3 – EXPLORATION OF PYTHON PACKAGES FOR DATA MINING TASK
AIM: To explore NumPy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-learn and Mlxtend using a small student dataset.
1) Check that the packages are installed
import numpy, pandas, matplotlib, seaborn
import scipy, sklearn, mlxtend
print("All required packages are installed.")
2) Create the dataset
import pandas as pd
data = {
    "Student_ID": ["S%02d" % i for i in range(1, 16)],
    "Study_Hours": [2,3,1,4,5,2,6,1,3,4,2,5,1,3,6],
    "Attendance": [65,70,55,80,85,60,90,50,75,82,68,88,52,72,92],
    "Internal_Mark": [48,55,40,65,72,45,80,35,58,68,50,75,38,56,82],
    "Assignment_Mark": [50,58,42,68,75,48,82,38,60,70,52,78,40,59,85],
    "Final_Mark": [52,60,45,70,78,50,85,40,63,73,55,80,42,61,88],
    "Result": ["Pass","Pass","Fail","Pass","Pass","Pass","Pass","Fail",
               "Pass","Pass","Pass","Pass","Fail","Pass","Pass"]
}
pd.DataFrame(data).to_csv("student_performance.csv", index=False)
print("Dataset created")
3) NumPy: Statistics of the final marks
import numpy as np
marks = np.array([52,60,45,70,78,50,85,40,63,73,55,80,42,61,88])
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Standard deviation:", round(np.std(marks), 2))
4) NumPy: Working on the whole array
print("Marks + 5 grace:", marks + 5)
print("Above 70:", marks[marks > 70])
print("Count above 70:", np.sum(marks > 70))
print("Sorted:", np.sort(marks))
5) Pandas: Load and look at the data
df = pd.read_csv("student_performance.csv")
print(df.head())
print("Shape:", df.shape)
6) Pandas: Statistical summary
print(df.describe().round(2))
7) Pandas: Missing values and duplicates
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
8) Pandas: Filter and group
high = df[df["Final_Mark"] >= 70]
print(high[["Student_ID", "Final_Mark", "Result"]])
print(df.groupby("Result")["Final_Mark"].mean())
9) Matplotlib: Histogram
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.hist(df["Final_Mark"], bins=5, edgecolor="black")
plt.xlabel("Final Mark")
plt.ylabel("Number of Students")
plt.title("Distribution of Final Marks")
plt.show()
10) Matplotlib: Scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(df["Study_Hours"], df["Final_Mark"], color="green")
plt.xlabel("Study Hours")
plt.ylabel("Final Mark")
plt.title("Study Hours vs Final Mark")
plt.grid(True)
plt.show()
11) Seaborn: Box plot
import seaborn as sns
plt.figure(figsize=(7, 5))
sns.boxplot(x="Result", y="Final_Mark", data=df)
plt.title("Final Mark by Result")
plt.show()
12) Seaborn: Correlation heatmap
num = df[["Study_Hours", "Attendance", "Internal_Mark",
         "Assignment_Mark", "Final_Mark"]]
plt.figure(figsize=(8, 6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Correlation Matrix")
plt.show()
13) SciPy: Correlation and z-scores
from scipy import stats
r, p = stats.pearsonr(df["Study_Hours"], df["Final_Mark"])
print("r =", round(r, 3), " p-value =", format(p, ".2e"))
z = stats.zscore(df["Final_Mark"])
print("z-scores:", np.round(z[:5], 2))
14) Scikit-learn: Split the data
from sklearn.model_selection import train_test_split
X = df[["Study_Hours", "Attendance",
        "Internal_Mark", "Assignment_Mark"]]
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
print("Training rows:", len(X_train), " Test rows:", len(X_test))
15) Scikit-learn: Train and evaluate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted:", y_pred.tolist())
print("Actual: ", y_test.tolist())
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=["Fail", "Pass"]))
16) Mlxtend: Frequent itemsets
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
baskets = [["Milk","Bread"], ["Milk","Bread","Butter"],
           ["Bread","Butter"], ["Milk","Bread","Butter"],
           ["Milk","Butter"]]
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(baskets).transform(baskets),
                      columns=te.columns_)
print(apriori(onehot, min_support=0.6, use_colnames=True))





EXPERIMENT 5 – IMPLEMENTATION OF THE APRIORI ALGORITHM
AIM: To implement Apriori in Python to find frequent itemsets and generate association rules from market-basket transactions.
1) Enter the transactions
transactions = [
    ["Milk", "Egg", "Bread", "Butter"],
    ["Milk", "Butter", "Egg", "ketchup"],
    ["Bread", "Butter", "ketchup"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Cookies"],
    ["Milk", "Bread", "Butter", "Cookies"],
    ["Milk", "Cookies"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Egg", "Cookies"],
    ["Milk", "Butter", "Bread"],
    ["Milk", "Bread", "Butter"],
    ["Milk", "Bread", "Cookie", "ketchup"]]
print(len(transactions), "transactions")
2) Preprocessing: Clean the item names
clean = []
for t in transactions:
    items = [i.strip().capitalize() for i in t]
    items = ["Cookies" if i == "Cookie" else i for i in items]
    clean.append(sorted(set(items)))
for n, t in enumerate(clean, 1):
    print(n, t)
3) One-hot encoding
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(clean).transform(clean),
                      columns=te.columns_)
print(onehot.astype(int))
4) Count each item
import matplotlib.pyplot as plt
counts = onehot.sum().sort_values(ascending=False)
print(counts)
counts.plot(kind="bar", color="steelblue", figsize=(7, 4.5))
plt.axhline(0.4 * len(onehot), color="red", linestyle="--")
plt.ylabel("Number of transactions")
plt.title("Item Counts")
plt.xticks(rotation=0)
plt.show()
5) Generate the frequent itemsets
from mlxtend.frequent_patterns import apriori
freq = apriori(onehot, min_support=0.4, use_colnames=True)
freq["itemsets"] = freq["itemsets"].apply(
    lambda s: ", ".join(sorted(s)))
print(freq.round(3))
6) Generate rules with lift >= 1
from mlxtend.frequent_patterns import association_rules
freq = apriori(onehot, min_support=0.4, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1)
show = rules[["antecedents", "consequents", "support",
              "confidence", "lift"]].copy()
for c in ["antecedents", "consequents"]:
    show[c] = show[c].apply(lambda s: ", ".join(sorted(s)))
print(show.round(3).to_string())
7) Rules by minimum confidence
r2 = association_rules(freq, metric="confidence", min_threshold=0.75)
show = r2[["antecedents", "consequents", "support",
            "confidence", "lift"]].copy()
for c in ["antecedents", "consequents"]:
    show[c] = show[c].apply(lambda s: ", ".join(sorted(s)))
print(show.round(3).to_string())





EXPERIMENT 10 – IMPLEMENTATION OF HIERARCHICAL AND DBSCAN CLUSTERING
AIM: To implement hierarchical (agglomerative) clustering with a dendrogram and DBSCAN density-based clustering, and compare DBSCAN with K-Means.
1) Install the packages (run in terminal/command prompt, not inside the Python script)
pip install numpy pandas matplotlib scipy scikit-learn
2) Part A – Hierarchical clustering: Enter the data points
import numpy as np
import matplotlib.pyplot as plt
X = np.array([(5,19), (4,17), (11,25), (6,22), (12,21),
              (10,21), (14,24), (3,16), (10,24), (4,21)])
for i, p in enumerate(X):
    print("Point", i, "=", tuple(int(v) for v in p))
3) Build the merge table
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X, method="average")
print(" merged       distance size")
for a, b, dist, size in Z:
    print("%3d and %-3d %7.2f %d" % (a, b, dist, size))
4) Draw the dendrogram
plt.figure(figsize=(8, 4.5))
dendrogram(Z)
plt.title("Dendrogram (average linkage)")
plt.xlabel("Point number")
plt.ylabel("Distance")
plt.show()
5) Form two clusters
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=2, linkage="average")
labels = model.fit_predict(X)
print("Cluster labels:", labels)
for c in [0, 1]:
    print("Cluster", c, ":", [tuple(int(v) for v in p)
                               for p in X[labels == c]])
6) Plot the clusters
plt.figure(figsize=(7, 4.5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", s=90)
for i, p in enumerate(X):
    plt.annotate(str(i), (p[0] + 0.15, p[1] + 0.15))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Agglomerative Clustering (2 clusters)")
plt.grid(True)
plt.show()
7) Part B – DBSCAN: Create the blob dataset
from sklearn.datasets import make_blobs
B, _ = make_blobs(n_samples=750, n_features=2, centers=3,
                  cluster_std=0.4, random_state=0)
print("Shape:", B.shape)
print(B[:3].round(2))
8) Run DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=4).fit(B)
lab = dbscan.labels_
n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
print("Clusters found:", n_clusters)
print("Noise points:", int((lab == -1).sum()))
print("Cluster sizes:", np.bincount(lab[lab >= 0]))
print("Core points:", len(dbscan.core_sample_indices_))
9) Plot the clusters
plt.figure(figsize=(7, 4.5))
plt.scatter(B[lab >= 0, 0], B[lab >= 0, 1], c=lab[lab >= 0],
            cmap="viridis", s=15)
plt.scatter(B[lab == -1, 0], B[lab == -1, 1], c="black",
            marker="x", s=60, label="Noise")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("DBSCAN (eps = 0.5, min_samples = 4)")
plt.legend()
plt.show()
10) Effect of eps and comparison on make_moons
for e in [0.1, 0.2, 0.3, 0.5, 1.0]:
    l = DBSCAN(eps=e, min_samples=4).fit(B).labels_
    k = len(set(l)) - (1 if -1 in l else 0)
    print("eps =", e, " clusters =", k,
          " noise =", int((l == -1).sum()))
 
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
M, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(M)
db = DBSCAN(eps=0.2, min_samples=4).fit_predict(M)
print("DBSCAN clusters:",
      len(set(db)) - (1 if -1 in db else 0),
      " noise:", int((db == -1).sum()))
11) Compare K-Means and DBSCAN results
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(M[:, 0], M[:, 1], c=km, cmap="coolwarm", s=15)
ax[0].set_title("K-Means (K = 2)")
ax[1].scatter(M[:, 0], M[:, 1], c=db, cmap="coolwarm", s=15)
ax[1].set_title("DBSCAN (eps = 0.2)")
plt.show()
 
# K-means cuts each moon in two, because it can only form round groups.
# DBSCAN follows the dense curves and recovers the two moons.