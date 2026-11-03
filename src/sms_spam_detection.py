"""
SMS Spam Detection: Comparison of 7 Machine Learning Algorithms
Research Implementation for:
'Comparison of Machine Learning Algorithms for Spam Detection with Multimodal Input Support'
APCIT 2024 / IEEE
"""

import os
import argparse
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


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load and clean the SMS Spam Collection dataset."""
    if not os.path.exists(dataset_path):
        # Check alternative common locations
        alt_paths = [
            os.path.join("dataset", "SMSSpamCollection"),
            os.path.join("..", "dataset", "SMSSpamCollection"),
            "SMSSpamCollection"
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                dataset_path = p
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Dataset not found at {dataset_path} or search paths.")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(
        dataset_path, sep="\t", header=None,
        names=["label", "message"], encoding="utf-8"
    )
    print(f"Raw dataset rows: {len(df)}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")

    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df["label"].isin(["ham", "spam"])].copy()

    print(f"Cleaned dataset shape: {df.shape}")
    print("Class distribution:\n", df["label"].value_counts())
    return df


def train_and_evaluate(df: pd.DataFrame, output_dir: str = "."):
    """Train 7 ML algorithms and evaluate their performance."""
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures") if output_dir != "." else "figures"
    results_dir = os.path.join(output_dir, "results") if output_dir != "." else "results"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20,
        random_state=42, stratify=df["label"]
    )

    # 2. TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2),
        min_df=2, stop_words="english"
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"\nTraining samples: {X_train_tfidf.shape[0]}")
    print(f"Testing samples: {X_test_tfidf.shape[0]}")
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # 3. Select KNN k using 3-fold cross-validation
    f1_spam = make_scorer(f1_score, pos_label="spam")
    knn_scores = {}
    for k in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_tfidf, y_train, cv=3, scoring=f1_spam)
        knn_scores[k] = scores.mean()

    best_k = max(knn_scores, key=knn_scores.get)
    print(f"\nKNN CV F1 scores: {knn_scores}")
    print(f"Selected KNN k: {best_k}")

    # 4. Define 7 ML Algorithms
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

    # 5. Train and Benchmark
    results = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training and evaluating: {name}...")
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
            "Precision": precision_score(y_test, pred, pos_label="spam", zero_division=0),
            "Recall": recall_score(y_test, pred, pos_label="spam", zero_division=0),
            "F1-Score": f1_score(y_test, pred, pos_label="spam", zero_division=0)
        })

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(results_df.round(4).to_string(index=False))

    best_name = results_df.iloc[0]["Algorithm"]
    best_pred = predictions[best_name]

    print(f"\nHighest test F1-score model: {best_name}")
    print("\n" + "=" * 50)
    print(f"CLASSIFICATION REPORT - {best_name}")
    print("=" * 50)
    print(classification_report(y_test, best_pred, labels=["ham", "spam"], zero_division=0))

    # 6. Save Comparison Plot
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plot_df = results_df.set_index("Algorithm")[metrics]
    ordered_algo = [
        "Random Forest", "SVM", "Naive Bayes", "Decision Tree",
        "Gradient Boosting", "Logistic Regression", "KNN"
    ]
    plot_df = plot_df.reindex(ordered_algo).dropna()

    x = np.arange(len(plot_df.index))
    width = 0.19

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - 1.5 * width, plot_df["Accuracy"], width, label="Accuracy")
    bars2 = ax.bar(x - 0.5 * width, plot_df["Precision"], width, label="Precision")
    bars3 = ax.bar(x + 0.5 * width, plot_df["Recall"], width, label="Recall")
    bars4 = ax.bar(x + 1.5 * width, plot_df["F1-Score"], width, label="F1-Score")

    ax.set_title(
        "Performance Comparison of Seven ML Algorithms for Spam Detection",
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
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)

    for bars in [bars1, bars2, bars3, bars4]:
        ax.bar_label(
            bars,
            labels=[f"{bar.get_height() * 100:.1f}%" for bar in bars],
            fontsize=7, rotation=90, padding=3
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    comparison_img_path = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(comparison_img_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, best_pred, labels=["ham", "spam"],
        display_labels=["Ham", "Spam"], cmap="Blues",
        values_format="d", ax=ax
    )
    ax.set_title(f"Confusion Matrix - {best_name}", fontweight="bold")
    plt.tight_layout()
    cm_img_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 8. Save CSV results
    csv_path = os.path.join(results_dir, "model_comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results saved to: {csv_path}")
    print(f"[OK] Comparison graph saved to: {comparison_img_path}")
    print(f"[OK] Confusion matrix saved to: {cm_img_path}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMS Spam Detection ML Comparison")
    parser.add_argument("--dataset", type=str, default="SMSSpamCollection", help="Path to SMSSpamCollection file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save figures and results")
    args = parser.parse_args()

    df_data = load_dataset(args.dataset)
    train_and_evaluate(df_data, args.output_dir)
