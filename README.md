# Kaggle NLP

This repository includes all scripts used for exploration and analysis of the [Kaggle NLP](https://www.kaggle.com/c/nlp-getting-started/code) project. We used several methods that are shown in the various scripts.

## Tweet Classifier

This is the main machine learning script. It produces a data frame that shows accuracies for the models and vectorizer methods we trained. The models are Logistic Regression, Random Forest, Support Vector Machine, and Naive Bayes. We applied both the Bag of Words and Tf-Idf vectorizers to each of these. Logistic Regression was the most accurate using the Tf-Idf vectorizer.

## Logit

Here, we explored cross validation on the Logistic Regression model to see if hyperparameter tuning would produce better results. It did not.

## Random Forest

We once again attempt to improve predictions through hyperparameter tuning. This time, we tuned the Random Forest Classifier and did not observe better predictions.

## Deep Classifier

This script implements word embeddings using neural networks. This also was not better than the base logit model.

## Submission

Using the knowledge from the previous scripts, we applied our Logistic Regression model to the entire training set, and used it to predict the test set given by Kaggle. This script produces the CSV that we submitted for 79% accuracy.
