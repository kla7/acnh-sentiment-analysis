# Sentiment Classification of Animal Crossing: New Horizons Reviews

**Author:** Kasey La (kaseyla@brandeis.edu)


This is my project for the final in NLP 1 (COSI 114a) where students have the space to apply NLP skills learned
throughout the semester. The tasks were to select a dataset; process the data; run experiments with different models,
features, and hyperparameters; present a brief summary of the project; and write a short paper about the process and
the findings.

This folder contains 6 files:

1. This **README** file.
2. **dataset.csv**, a file containing 2999 reviews with sentiment labels (negative, neutral, positive).
3. **project.py**, a script that loads and preprocesses data, trains models, and prints evaluation scores.
4. **presentation.pptx**, presentation slides that contain a brief summary of the data and modeling.
5. **writeup.pdf**, a detailed documentation of the data, the models and feature sets used, development set results,
   and test set results.
6. **accuracy_scores.md**, a file containing the accuracy scores of all hyperparameters ran per model during the
   experimental phase.

### Overview of data in _dataset.csv_
The data was retrieved from
[this Animal Crossing reviews dataset on Kaggle](https://www.kaggle.com/datasets/jessemostipak/animal-crossing)
which retrieved reviews from Metacritic in 2020. The original dataset contained a raw score from 0-10 (where a higher
score corresponds to a better review), the username of the reviewer, the raw text of the review, and the date that the
review was posted. Only the raw scores and raw text were kept for this project. Those raw scores were binned into
the three sentiment labels: 0-3 for Negative, 4-6 for Neutral, and 7-10 for Positive. After binning, the number of
reviews per sentiment label are as follows: 1642 Negative, 227 Neutral, and 1130 Positive.

The table below includes a few examples from the dataset:

| **Label** | **Text**                                                                                                                                   |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Negative  | This is so annoying. Only one player has the ability to play this game because only one island. Nintendo doesn't deserve my money anymore. |
| Negative  | One island per switch. I have to let my little sister play and i have to stay player 2                                                     |
| Neutral   | The game is fun to play, but once again Nintendo fumbles online which is why my review is a 5.                                             |
| Neutral   | The 2 players experience is terrible. I would recommand this game to a person with no friends, or family.                                  |
| Positive  | This game is absolutely amazing as a constant side game for one's day. It a great game to relax with.                                      |
| Positive  | Super fun game. I do question Tom Nook's predatory lending practices. I owe him more than I do my actual bank.                             |

Each line in this file contains the following information:

- **Label**: The label assigned to the review after binning the raw score.
- **Text**: The unedited text contents of the review.

### Overview of _project.py_

This script utilizes sklearn models Multinomial Naive Bayes, Logistic Regression, and Random Forest Classifier to run
experiments on two feature sets. The first feature set contains lowercased unigrams with stop words, punctuation, and
spaces removed. The second feature set contains lowercased bigrams with punctuation and spaces removed. Both feature
sets were experimented on with the development set and only the unigrams feature set was used for the test set.

The list of hyperparameters tuned per model are listed below:
- **Multinomial Naive Bayes** - alpha (α)
- **Logistic Regression** -  C, class_weight, max_iter
- **Random Forest Classifier** - n_estimators, max_depth, max_features