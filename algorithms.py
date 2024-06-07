import itertools
import json as js
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from typing import Iterator

# Load datasets
df1: pd.DataFrame = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/project/data/Cardiovascular_Disease_Dataset_final.csv")
df2: pd.DataFrame = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/project/data/dataset_37_diabetes_final.csv")
df3: pd.DataFrame = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/project/data/diabetes_classification_final.csv")
df4: pd.DataFrame = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/project/data/Medicaldataset_final.csv")

# Define datasets
datasets: dict[str, pd.DataFrame] = {
    "Cardiovascular_Disease_Dataset": df1,
    "Diabetes_Dataset_37": df2,
    "Diabetes_Classification": df3,
    "Medical_Dataset": df4
}

algorithms: list[BaseEstimator] = [LogisticRegression, RandomForestClassifier, GaussianNB, DecisionTreeClassifier, KNeighborsClassifier]

result: dict[str, dict[str, dict[str, float]]] = {}

def get_combination(lst: list[str]) -> Iterator[tuple[str]]:
    combination: list[tuple[str]] = []
    for r in range(1, len(lst) + 1):
        combination.extend(itertools.combinations(lst,r))
    yield from combination

def evaluate_algorithm(algorithm) -> None:
    algorithm_name: str = algorithm.__name__
    result[algorithm_name] = {}
    
    dataset_name: str
    df: pd.DataFrame
    for dataset_name, df in datasets.items():

        columns: list[str] = list(df.columns)

        result[algorithm_name][dataset_name] = {}

        for combination in get_combination(columns[:-1]):
            combination = sorted(combination)
            print(algorithm_name, combination)
            X = df.loc[:,combination]    
            y = df.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model: BaseEstimator = algorithm(max_iter=10000) if algorithm == LogisticRegression else algorithm()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy: float = accuracy_score(y_test, y_pred)
            
            result[algorithm_name][dataset_name][", ".join(combination)] = accuracy

for algorithm in algorithms:
    evaluate_algorithm(algorithm)

json_object = js.dumps(result, indent=4)
with open("results.json", "w") as outfile:
    outfile.write(json_object)