# SMS Spam Detection: Comparison of 7 ML Algorithms
# Research implementation for APCIT 2024 / IEEE

import os
import sys

# Support both Google Colab and standard local Python
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, classification_report, make_scorer
)

# 1. Dataset loading
if IN_COLAB:
    print("Upload the extracted SMSSpamCollection file")
    uploaded = files.upload()
    filename = next(iter(uploaded))
else:
    if os.path.exists("SMSSpamCollection"):
        filename = "SMSSpamCollection"
    elif os.path.exists(os.path.join("dataset", "SMSSpamCollection")):
        filename = os.path.join("dataset", "SMSSpamCollection")
    elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        filename = sys.argv[1]
    else:
        filename = "SMSSpamCollection"
    print(f"Using dataset: {filename}")

# 2. Read and clean dataset
df = pd.read_csv(
    filename, sep="\t", header=None,
    names=["label", "message"], encoding="utf-8"
)
print("\nRaw dataset rows:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())

df = df.dropna(subset=["label", "message"])
df = df.drop_duplicates().reset_index(drop=True)
df = df[df["label"].isin(["ham", "spam"])].copy()

print("\nDataset shape after cleaning:", df.shape)
print("\nClass distribution:\n", df["label"].value_counts())
print("\nFirst 5 records:\n", df.head())

# 3. Split data (80:20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.20,
    random_state=42, stratify=df["label"]
)

# 4. TF-IDF feature extraction
tfidf = TfidfVectorizer(
    max_features=3000, ngram_range=(1, 2),
    min_df=2, stop_words="english"
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTraining samples:", X_train_tfidf.shape[0])
print("Testing samples:", X_test_tfidf.shape[0])
print("TF-IDF features:", X_train_tfidf.shape[1])

# 5. Select KNN k using training-set cross-validation
f1_spam = make_scorer(f1_score, pos_label="spam")
knn_scores = {}

for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn, X_train_tfidf, y_train, cv=3, scoring=f1_spam
    )
    knn_scores[k] = scores.mean()

best_k = max(knn_scores, key=knn_scores.get)
print("\nKNN CV F1 scores:", knn_scores)
print("Selected KNN k:", best_k)

# 6. Define seven algorithms
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "SVM": LinearSVC(C=1.0, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=best_k),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=3, random_state=42
    )
}

# 7. Train and evaluate models
results = []
predictions = {}
trained_models = {}

for name, model in models.items():
    print("\nTraining:", name)

    if name == "Gradient Boosting":
        Xtr = X_train_tfidf.toarray()
        Xte = X_test_tfidf.toarray()
    else:
        Xtr = X_train_tfidf
        Xte = X_test_tfidf

    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    predictions[name] = pred
    trained_models[name] = model

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, pos_label="spam", zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, pos_label="spam", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, pred, pos_label="spam", zero_division=0
        )
    })

# 8. Comparison table
results_df = pd.DataFrame(results).sort_values(
    "F1-Score", ascending=False
).reset_index(drop=True)

print("\n===== MODEL COMPARISON =====")
print(results_df.round(4).to_string(index=False))

best_name = results_df.iloc[0]["Algorithm"]
best_pred = predictions[best_name]

print("\nHighest test F1-score model:", best_name)
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test, best_pred, labels=["ham", "spam"], zero_division=0
))

# 9. Professional comparison graph
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
plot_df = results_df.set_index("Algorithm")[metrics]
plot_df = plot_df.loc[
    ["Random Forest", "SVM", "Naive Bayes", "Decision Tree",
     "Gradient Boosting", "Logistic Regression", "KNN"]
]

x = np.arange(len(plot_df.index))
width = 0.19

fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - 1.5 * width, plot_df["Accuracy"],
               width, label="Accuracy")
bars2 = ax.bar(x - 0.5 * width, plot_df["Precision"],
               width, label="Precision")
bars3 = ax.bar(x + 0.5 * width, plot_df["Recall"],
               width, label="Recall")
bars4 = ax.bar(x + 1.5 * width, plot_df["F1-Score"],
               width, label="F1-Score")

ax.set_title(
    "Performance Comparison of Seven ML Algorithms",
    fontsize=15, fontweight="bold", pad=18
)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_xlabel("Machine Learning Algorithms", fontsize=11)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=20, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(
    ncols=4, loc="upper center",
    bbox_to_anchor=(0.5, 1.0), frameon=False
)

for bars in [bars1, bars2, bars3, bars4]:
    ax.bar_label(
        bars,
        labels=[f"{bar.get_height() * 100:.1f}%"
                for bar in bars],
        fontsize=7, rotation=90, padding=3
    )

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. Confusion matrix of highest-F1 model
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, labels=["ham", "spam"],
    display_labels=["Ham", "Spam"], cmap="Blues",
    values_format="d", ax=ax
)
ax.set_title(f"Confusion Matrix - {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# 11. Save research results
results_df.to_csv("model_comparison_results.csv", index=False)
print("\nResults and figures saved successfully.")

# 12. Colab Download support (if running in Google Colab)
if IN_COLAB:
    files.download("model_comparison_results.csv")
    files.download("model_comparison.png")
    files.download("confusion_matrix.png")
