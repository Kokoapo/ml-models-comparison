# Machine Learning Models Comparisson

This Python project aims to compare different different machine learning models classifying penguins species.

## Running

To run this project, install the [dependencies](requirements.txt) with:

```Bash
pip install -r requirements.txt
```

## Dataset

The dataset used for this project is called [Penguin Species Dataset](https://www.kaggle.com/datasets/srisyra02/penguin-species-dataset), available at:

> srisyra02/penguin-species-dataset

## Models

There are, in total, 10 approaches used in this project:

| Model Name        | Type                      |
| :---              | :----:                    |
| KNN               | Monolithic                |
| Decision Tree     | Monolithic                | 
| Naive Bayes       | Monolithic                |
| SVM               | Monolithic                |
| MLP               | Monolithic                |
| Random Forest     | Multiple Classifiers      |
| Bagging           | Multiple Classifiers      |
| Boosting          | Multiple Classifiers      |
| Sum Rule          | Multiple Classifiers      |
| Majoritary Voting | Multiple Classifiers      |

It is important to notice that both **Sum Rule** and **Majoritary Voting** uses all the Monolithic methods as estimators