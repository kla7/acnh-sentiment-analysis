# Accuracy Scores

This file contains tables of all accuracy scores for the models and hyperparameters tuned per feature set and the
dataset ran. Summarized versions of each table can be found in **writeup.pdf**.

### Development set - unigrams feature set

| **Model**           | **Hyperparameters**                                                           | Accuracy |
|---------------------|-------------------------------------------------------------------------------|----------|
| Baseline            | N/A                                                                           | 55.00    |
| Naive Bayes         | alpha = 0.5                                                                   | 82.00    |
| Naive Bayes         | alpha = 1<br/>(DEFAULT)                                                       | 82.33    |
| Naive Bayes         | alpha = 1.1                                                                   | 83.00    |
| Naive Bayes         | alpha = 1.2                                                                   | 83.00    |
| Naive Bayes         | alpha = 1.3                                                                   | 83.33    |
| Naive Bayes         | alpha = 1.35                                                                  | 83.67    |
| Naive Bayes         | alpha = 1.36                                                                  | 84.00    |
| Naive Bayes         | alpha = 1.37                                                                  | 84.00    |
| Naive Bayes         | alpha = 1.38                                                                  | 84.00    |
| Naive Bayes         | alpha = 1.39                                                                  | 84.00    |
| Naive Bayes         | alpha = 1.4                                                                   | 84.00    |
| Naive Bayes         | alpha = 1.42                                                                  | 84.00    |
| Naive Bayes         | alpha = 1.45                                                                  | 84.00    |
| Naive Bayes         | alpha = 1.5                                                                   | 83.33    |
| Logistic Regression | c = 1.0<br/>max_iter = 100<br/>class_weight = None<br/>(DEFAULT)              | 84.33    |
| Logistic Regression | max_iter = 4000                                                               | 85.00    |
| Logistic Regression | class_weight = balanced                                                       | 82.67    |
| Logistic Regression | max_iter = 4000<br/>class_weight = balanced                                   | 83.00    |
| Logistic Regression | c = 0.01                                                                      | 84.00    |
| Logistic Regression | c = 0.01<br/>max_iter = 4000                                                  | 84.00    |
| Logistic Regression | c = 0.01<br/>class_weight = balanced                                          | 79.33    |
| Logistic Regression | c = 0.01<br/>max_iter = 4000<br/>class_weight = balanced                      | 79.33    |
| Logistic Regression | c = 0.1                                                                       | 86.00    |
| Logistic Regression | c = 0.1<br/>max_iter = 4000                                                   | 86.33    |
| Logistic Regression | c = 0.1<br/>class_weight = balanced                                           | 83.67    |
| Logistic Regression | c = 0.1<br/>max_iter = 4000<br/>class_weight = balanced                       | 84.00    |
| Logistic Regression | c = 0.5                                                                       | 85.00    |
| Logistic Regression | c = 0.5<br/>max_iter = 4000                                                   | 85.00    |
| Logistic Regression | c = 0.5<br/>class_weight = balanced                                           | 82.67    |
| Logistic Regression | c = 0.5<br/>max_iter = 4000<br/>class_weight = balanced                       | 83.00    |
| Logistic Regression | c = 2.0                                                                       | 84.33    |
| Logistic Regression | c = 2.0<br/>max_iter = 4000                                                   | 84.00    |
| Logistic Regression | c = 2.0<br/>class_weight = balanced                                           | 82.67    |
| Logistic Regression | c = 2.0<br/>max_iter = 4000<br/>class_weight = balanced                       | 83.00    |
| Random Forest       | max_depth = None<br/>n_estimators = 100<br/>max_features = sqrt<br/>(DEFAULT) | 82.67    |
| Random Forest       | max_features = log2                                                           | 83.00    |
| Random Forest       | n_estimators = 50                                                             | 81.00    |
| Random Forest       | n_estimators = 150                                                            | 82.33    |
| Random Forest       | n_estimators = 150<br/>max_features = log2                                    | 81.67    |
| Random Forest       | n_estimators = 200                                                            | 82.00    |
| Random Forest       | n_estimators = 200<br/>max_features = log2                                    | 81.33    |
| Random Forest       | n_estimators = 400                                                            | 82.33    |
| Random Forest       | max_depth = 1                                                                 | 55.00    |
| Random Forest       | max_depth = 3                                                                 | 55.33    |
| Random Forest       | max_depth = 5                                                                 | 58.67    |
| Random Forest       | max_depth = 7                                                                 | 72.00    |
| Random Forest       | max_depth = 9                                                                 | 75.33    |
| Random Forest       | max_depth = 9<br/>n_estimators = 50                                           | 74.33    |
| Random Forest       | max_depth = 9<br/>n_estimators = 150                                          | 75.33    |
| Random Forest       | max_depth = 9<br/>n_estimators = 200                                          | 75.33    |

### Development set - bigrams feature set

| **Model**           | **Hyperparameters**                                                           | Accuracy |
|---------------------|-------------------------------------------------------------------------------|----------|
| Baseline            | N/A                                                                           | 55.00    |
| Naive Bayes         | alpha = 1<br/>(DEFAULT)                                                       | 81.33    |
| Naive Bayes         | alpha = 1.35                                                                  | 81.33    |
| Logistic Regression | c = 1.0<br/>max_iter = 100<br/>class_weight = None<br/>(DEFAULT)              | 82.67    |
| Logistic Regression | c = 0.1<br/>max_iter = 4000                                                   | 83.00    |
| Logistic Regression | c = 0.1<br/>max_iter = 4000<br/>class_weight = balanced                       | 81.00    |
| Logistic Regression | c = 0.5<br/>max_iter = 4000                                                   | 82.67    |
| Random Forest       | max_depth = None<br/>n_estimators = 100<br/>max_features = sqrt<br/>(DEFAULT) | 79.33    |
| Random Forest       | n_estimators = 150                                                            | 82.00    |
| Random Forest       | n_estimators = 400                                                            | 82.33    |

### Test set - unigrams feature set

| **Model**           | **Hyperparameters**                                     | Accuracy |
|---------------------|---------------------------------------------------------|----------|
| Logistic Regression | c = 0.1<br/>max_iter = 4000                             | 84.33    |
| Logistic Regression | c = 0.1<br/>max_iter = 4000<br/>class_weight = balanced | 81.33    |
| Logistic Regression | c = 0.5<br/>max_iter = 4000                             | 83.00    |