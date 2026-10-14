KNN = {
    'n_neighbors': 3,              # [3, 7, 11],
    'weights': 'distance'          # ['uniform', 'distance']
}

AD = {
    'criterion': 'gini',            # ['gini', 'entropy', 'log_loss'],
    'splitter': 'best',             # ['best', 'random'],
    'max_depth': 5,                 # [None, 5, 10],
    'min_samples_split': 10,        # [2, 10, 50],
    'min_samples_leaf': 1           # [1, 4, 16]
}

NB = {}

SVM = {
    'C': 1.0,                      # [0.01, 0.1, 1.0, 10, 100],
    'kernel': 'linear',            # ['linear', 'rbf', 'poly', 'sigmoid']
    'probability': True
}

MLP = {
    'hidden_layer_sizes': (100,),    # [(50,), (100,), (50,50,), (100,100,), (10,10,), (10,10,10,), (50, 50, 50, 50, 50,)],
    'activation': 'relu',           # ['relu', 'tanh', 'logistic'],
    'batch_size': 32,               # [16, 32, 64],
    'max_iter': 500,                # [200, 500, 1000],
    'learning_rate_init': 0.0001     # [0,0001, 0.001, 0.01, 0.1],
}

RF = {
    'n_estimators': 100,             # [50, 100, 200],
    'criterion': 'entropy',            # ['gini', 'entropy'],
    'max_depth': 10,              # [None, 5, 10, 20, 30],
    'min_samples_split': 2,         # [2, 5, 10],
    'min_samples_leaf': 4           # [1, 2, 4]
}

BAGGING = {
    'n_estimators': 100,             # [10, 50, 100],
    'max_samples': 0.7              # [0.5, 0.7, 1.0]
}

BOOSTING = {
    'n_estimators': 200,             # [50, 100, 200],
    'learning_rate': 0.5           # [0.01, 0.1, 0.5]
}