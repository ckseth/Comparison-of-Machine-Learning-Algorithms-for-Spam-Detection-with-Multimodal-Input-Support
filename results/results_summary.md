# Experimental Evaluation and Model Comparison Results

Evaluation of Seven Machine Learning Algorithms on the UCI SMS Spam Collection Dataset (80:20 Stratified Split, TF-IDF Feature Representation).

## Benchmark Results Table

| Rank | Algorithm | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Random Forest** | **98.07%** | **100.00%** | 84.73% | **91.74%** |
| 2 | **Support Vector Machine (SVM)** | 97.87% | 97.39% | 85.50% | 91.06% |
| 3 | **Multinomial Naive Bayes** | 97.49% | 100.00% | 80.15% | 88.98% |
| 4 | **Decision Tree** | 97.00% | 89.68% | **86.26%** | 87.94% |
| 5 | **Gradient Boosting** | 96.23% | 100.00% | 70.23% | 82.51% |
| 6 | **Logistic Regression** | 95.94% | 97.85% | 69.47% | 81.25% |
| 7 | **K-Nearest Neighbors (KNN)** | 92.94% | 100.00% | 44.27% | 61.38% |

## Key Findings

1. **Random Forest** achieved the highest overall F1-score (**91.74%**) and accuracy (**98.07%**) with a perfect 100% precision on spam messages (zero false positives).
2. **Support Vector Machine (LinearSVC)** was the closest competitor with an F1-score of **91.06%** and balanced recall (**85.50%**).
3. **Naive Bayes, Gradient Boosting, and KNN** maintained **100.00% Precision**, ensuring that legitimate emails/SMS (`ham`) are virtually never flagged as spam by mistake.
4. The Confusion Matrix demonstrates minimal misclassifications under the best-performing ensemble architecture.

