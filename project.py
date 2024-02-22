import pandas as pd
import spacy
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Feature set 1: unigrams (lowercased), with no stop words, punctuation, or spaces
def run_feature_set1(dataset, model_type=None, dev_or_test="dev", is_baseline=False, **kwargs):
    all_unigrams = []

    for doc in dataset["doc"]:
        unigrams = extract_words(doc)
        all_unigrams.append(Counter(unigrams))

    vectorizer = DictVectorizer()

    fit_transform = vectorizer.fit_transform(all_unigrams)
    processed_df = pd.DataFrame.sparse.from_spmatrix(fit_transform)
    processed_df.columns = vectorizer.vocabulary_
    processed_df["label"] = dataset["label"]

    train_df, test_df = train_test_split(processed_df, test_size=.2, train_size=.8, random_state=77)
    test_df, dev_df = train_test_split(test_df, test_size=.5, train_size=.5, random_state=77)

    train_y = train_df["label"]
    train_x = train_df.drop(columns="label")

    test_y = test_df["label"]
    test_x = test_df.drop(columns="label")

    dev_y = dev_df["label"]
    dev_x = dev_df.drop(columns="label")

    if is_baseline:
        most_frequent_label = train_y.value_counts().index[0]
        predicted_dev_y = [most_frequent_label] * len(dev_y)
        predicted_test_y = [most_frequent_label] * len(test_y)
    else:
        classifier = model_type(**kwargs)
        classifier.fit(train_x, train_y)

        predicted_dev_y = classifier.predict(dev_x)
        predicted_test_y = classifier.predict(test_x)

    if dev_or_test == "dev":
        report = classification_report(y_true=dev_y, y_pred=predicted_dev_y, zero_division=0.0, digits=4)
        print(f"Dev set report:\n{report}\n")
    else:
        report = classification_report(y_true=test_y, y_pred=predicted_test_y, zero_division=0.0, digits=4)
        print(f"Test set report:\n{report}\n")


# Feature set 2: bigrams (lowercased), no punctuation or spaces
def run_feature_set2(dataset, model_type=None, dev_or_test="dev", is_baseline=False, **kwargs):
    all_bigrams = []

    for doc in dataset["doc"]:
        bigrams = []
        words = extract_words(doc, with_stop_words=True)

        for index in range(len(words)-1):
            bigrams.append(f"({words[index]}, {words[index + 1]})")

        all_bigrams.append(Counter(bigrams))

    vectorizer = DictVectorizer()

    fit_transform = vectorizer.fit_transform(all_bigrams)
    processed_df = pd.DataFrame.sparse.from_spmatrix(fit_transform)
    processed_df.columns = vectorizer.vocabulary_
    processed_df["label"] = dataset["label"]

    train_df, test_df = train_test_split(processed_df, test_size=.2, train_size=.8, random_state=77)
    test_df, dev_df = train_test_split(test_df, test_size=.5, train_size=.5, random_state=77)

    train_y = train_df["label"]
    train_x = train_df.drop(columns="label")

    test_y = test_df["label"]
    test_x = test_df.drop(columns="label")

    dev_y = dev_df["label"]
    dev_x = dev_df.drop(columns="label")

    if is_baseline:
        most_frequent_label = train_y.value_counts().index[0]
        predicted_dev_y = [most_frequent_label] * len(dev_y)
        predicted_test_y = [most_frequent_label] * len(test_y)
    else:
        classifier = model_type(**kwargs)
        classifier.fit(train_x, train_y)

        predicted_dev_y = classifier.predict(dev_x)
        predicted_test_y = classifier.predict(test_x)

    if dev_or_test == "dev":
        report = classification_report(y_true=dev_y, y_pred=predicted_dev_y, zero_division=0.0, digits=4)
        print(f"Dev set report:\n{report}\n")
    else:
        report = classification_report(y_true=test_y, y_pred=predicted_test_y, zero_division=0.0, digits=4)
        print(f"Test set report:\n{report}\n")


# Extracts words from the document with an option to include or exclude stop words
def extract_words(doc, with_stop_words=True):
    word_list = []

    if with_stop_words:
        for t in doc:
            if t.is_space or t.is_punct:
                continue
            word_list.append(t.lower_)
    else:
        if with_stop_words:
            for t in doc:
                if t.is_space or t.is_punct or t.is_stop:
                    continue
                word_list.append(t.lower_)

    return word_list


# Pre-process data
nlp = spacy.load("en_core_web_sm")
df = pd.read_csv("dataset.csv")
df["doc"] = df["text"].apply(nlp)

# Configurations to run on unigram development set
unigram_dev_configs = {MultinomialNB:
                       [{"alpha": 0.5}, {}, {"alpha": 1.1}, {"alpha": 1.2}, {"alpha": 1.3},
                        {"alpha": 1.35}, {"alpha": 1.36}, {"alpha": 1.37}, {"alpha": 1.38}, {"alpha": 1.39},
                        {"alpha": 1.4}, {"alpha": 1.42}, {"alpha": 1.45}, {"alpha": 1.5}],
                       LogisticRegression:
                       [{}, {"max_iter": 4000}, {"class_weight": "balanced"},
                        {"max_iter": 4000, "class_weight": "balanced"},
                        {"C": 0.01}, {"C": 0.01, "max_iter": 4000}, {"C": 0.01, "class_weight": "balanced"},
                        {"C": 0.01, "max_iter": 4000, "class_weight": "balanced"},
                        {"C": 0.1}, {"C": 0.1, "max_iter": 4000}, {"C": 0.1, "class_weight": "balanced"},
                        {"C": 0.1, "max_iter": 4000, "class_weight": "balanced"},
                        {"C": 0.5}, {"C": 0.5, "max_iter": 4000}, {"C": 0.5, "class_weight": "balanced"},
                        {"C": 0.5, "max_iter": 4000, "class_weight": "balanced"},
                        {"C": 2.0}, {"C": 2.0, "max_iter": 4000}, {"C": 2.0, "class_weight": "balanced"},
                        {"C": 2.0, "max_iter": 4000, "class_weight": "balanced"}],
                       RandomForestClassifier:
                       [{}, {"max_features": "log2"}, {"n_estimators": 50}, {"n_estimators": 150},
                        {"n_estimators": 150, "max_features": "log2"}, {"n_estimators": 200},
                        {"n_estimators": 200, "max_features": "log2"}, {"n_estimators": 400},
                        {"max_depth": 1}, {"max_depth": 3}, {"max_depth": 5}, {"max_depth": 7},
                        {"max_depth": 9}, {"max_depth": 9, "n_estimators": 50},
                        {"max_depth": 9, "n_estimators": 150}, {"max_depth": 9, "n_estimators": 200}]}

# Configurations to run on bigram development set
bigram_dev_configs = {MultinomialNB:
                      [{}, {"alpha": 1.35}],
                      LogisticRegression:
                      [{}, {"C": 0.1, "max_iter": 4000}, {"C": 0.1, "max_iter": 4000, "class_weight": "balanced"},
                       {"C": 0.5, "max_iter": 4000}],
                      RandomForestClassifier:
                      [{}, {"n_estimators": 150}, {"n_estimators": 400}]}

# Configurations to run on unigram test set
unigram_test_configs = {LogisticRegression:
                        [{"C": 0.1, "max_iter": 4000}, {"C": 0.1, "max_iter": 4000, "class_weight": "balanced"},
                         {"C": 0.5, "max_iter": 4000}]}

# Run unigram baseline experiment (most frequent class)
print("UNIGRAM BASELINE EXPERIMENT")
run_feature_set1(df, is_baseline=True)

# Run unigram development set experiments
print("UNIGRAM DEV SET EXPERIMENTS")
for model, parameter_list in unigram_dev_configs.items():
    for parameter in parameter_list:
        print(f"{model.__name__} - {parameter}")
        run_feature_set1(df, model, **parameter)

# Run bigram baseline experiment (most frequent class)
print("BIGRAM BASELINE EXPERIMENT")
run_feature_set2(df, is_baseline=True)

# Run bigram development set experiments
print("BIGRAM DEV SET EXPERIMENTS")
for model, parameter_list in bigram_dev_configs.items():
    for parameter in parameter_list:
        print(f"{model.__name__} - {parameter}")
        run_feature_set2(df, model, **parameter)

# Run unigram test set experiments
print("UNIGRAM TEST SET EXPERIMENTS")
for model, parameter_list in unigram_test_configs.items():
    for parameter in parameter_list:
        print(f"{model.__name__} - {parameter}")
        run_feature_set1(df, model, dev_or_test="test", **parameter)
