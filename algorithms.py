import itertools
import json as js
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from typing import Iterator, List, Dict, Tuple

"""Loading datasets"""
df1: pd.DataFrame = pd.read_csv("data/Cardiovascular_Disease_Dataset_final.csv")
df2: pd.DataFrame = pd.read_csv("data/dataset_37_diabetes_final.csv")
df3: pd.DataFrame = pd.read_csv("data/diabetes_classification_final.csv")
df4: pd.DataFrame = pd.read_csv("data/Medicaldataset_final.csv")

datasets: Dict[str, pd.DataFrame] = {
    "Cardiovascular_Disease_Dataset": df1,
    "Diabetes_Dataset_37": df2, 
    "Diabetes_Classification": df3,
    "Medical_Dataset": df4
}

algorithms: List[BaseEstimator] = [LogisticRegression, RandomForestClassifier, GaussianNB, DecisionTreeClassifier, KNeighborsClassifier]
"""Algorithms to be compared"""
result: Dict[str, Dict[str, Dict[str, float]]] = {}

def get_combination(columns: List[str]) -> Iterator[Tuple[str]]:
    """Yields all possible combinations of columns"""

    combination: List[Tuple[str]] = []
    for r in range(1, len(columns) + 1):
        combination.extend(itertools.combinations(columns, r))
    yield from combination

def evaluate_algorithm(algorithm) -> None:
    """Evaluate the specified machine learning algorithm on each dataset using all possible combinations of features.
       Evaluation metrics: Accuracy score, Precision score, Recall score, F1 score, ROC AUC"""
    
    algorithm_name = algorithm.__name__
    result[algorithm_name] = {}
    
    dataset_name: str
    df: pd.DataFrame
    for dataset_name, df in datasets.items():
        columns = list(df.columns)
        result[algorithm_name][dataset_name] = {}

        for combination in get_combination(columns[:-1]):
            combination = sorted(combination)
            X: pd.DataFrame = df.loc[:, combination]    
            y: pd.DataFrame = df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = algorithm(max_iter=10000) if algorithm == LogisticRegression else algorithm()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            accuracy: float = accuracy_score(y_test, y_pred)
            precision: float = precision_score(y_test, y_pred)
            recall: float = recall_score(y_test, y_pred)
            f1: float = f1_score(y_test, y_pred)
            roc_auc: float = roc_auc_score(y_test, y_prob) if y_prob is not None else None

            result[algorithm_name][dataset_name][", ".join(combination)] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }

algorithm: BaseEstimator
for algorithm in algorithms:
    evaluate_algorithm(algorithm)
"""Evaluate each algorithm"""

json_object = js.dumps(result, indent=4)
with open("results.json", "w") as outfile:
    outfile.write(json_object)
"""The result dictionary is written into JSON file"""

