from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import kagglehub, os, hyperparameters as hyperparameters
import pandas as pd

MODELS = {
    "KNN": KNeighborsClassifier,
    "AD": DecisionTreeClassifier,
    "NB": GaussianNB,
    "SVM": SVC,
    "MLP": MLPClassifier,
    "RF": RandomForestClassifier,
    "Bagging": BaggingClassifier,
    "Boosting": AdaBoostClassifier
}

def preprocess(): 
    path = kagglehub.dataset_download("srisyra02/penguin-species-dataset")
    df = pd.read_csv(os.path.join(path, 'Penguin Species Prediction Dataset.csv'))

    df = df.drop(columns=['island', 'sex'])
    df = df.dropna()

    X = df.drop(columns=['species'])
    y = df['species']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)
    # print("[PreProcess] Dataframe Splitted")
    # print(f"\tTrain:\t{len(X_train)} samples ({len(X_train)/len(X):.1%})")
    # print(f"\tVal.:\t{len(X_val)} samples ({len(X_val)/len(X):.1%})")
    # print(f"\tTest:\t{len(X_test)} samples ({len(X_test)/len(X):.1%})")
    # print("[PreProcess] Class Proportion in Each Set")
    # print(f"\tFirst:\t{y.value_counts(normalize=True).round(3).to_dict()}")
    # print(f"\tTrain:\t{y_train.value_counts(normalize=True).round(3).to_dict()}")
    # print(f"\tVal.:\t{y_val.value_counts(normalize=True).round(3).to_dict()}")
    # print(f"\tTest:\t{y_test.value_counts(normalize=True).round(3).to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def setup_model(name, **kwargs):
    return MODELS[name](**kwargs)

if __name__ == "__main__":
    results = []
    for _ in range(20):
        models = {
            "KNN": setup_model("KNN", **hyperparameters.KNN),
            "AD": setup_model("AD", **hyperparameters.AD),
            "NB": setup_model("NB", **hyperparameters.NB),
            "SVM": setup_model("SVM", **hyperparameters.SVM),
            "MLP": setup_model("MLP", **hyperparameters.MLP),
            "RF": setup_model("RF", **hyperparameters.RF),
            "Bagging": setup_model("Bagging", **hyperparameters.BAGGING),
            "Boosting": setup_model("Boosting", **hyperparameters.BOOSTING),
            "VM": VotingClassifier(
                estimators=[
                    ("KNN", setup_model("KNN", **hyperparameters.KNN)),
                    ("AD", setup_model("AD", **hyperparameters.AD)),
                    ("NB", setup_model("NB", **hyperparameters.NB)),
                    ("SVM", setup_model("SVM", **hyperparameters.SVM)),
                    ("MLP", setup_model("MLP", **hyperparameters.MLP))
                ],
                voting='hard'
            ),
            "RS": VotingClassifier(
                estimators=[
                    ("KNN", setup_model("KNN", **hyperparameters.KNN)),
                    ("AD", setup_model("AD", **hyperparameters.AD)),
                    ("NB", setup_model("NB", **hyperparameters.NB)),
                    ("SVM", setup_model("SVM", **hyperparameters.SVM)),
                    ("MLP", setup_model("MLP", **hyperparameters.MLP))
                ],
                voting='soft'
            )
        }
        for name, model in models.items():
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess()
        
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

            val_acc = accuracy_score(y_val, y_val_pred)
            results.append({
                "model": name,
                "val_acc": val_acc,
            })
    df_results = pd.DataFrame(results).pivot_table(columns="model", values="val_acc", aggfunc="mean")
    print(df_results.head())

