from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Union, List, Dict
from sklearn.base import BaseEstimator

app = Flask(__name__)

datasets: Dict[str, Dict[str, Any]] = {
    "Cardiovascular diseases (India)": {
        "data": pd.read_csv("data/Cardiovascular_Disease_Dataset_final.csv"),
        "description": """The source of the data: https://data.mendeley.com/datasets/dzz48mvjht/1 \n
        This heart disease dataset is sourced from a multispecialty hospital in India.
        It includes data from 1,000 subjects, encompassing 12 key features relevant to heart disease.
        The dataset is designed for research purposes, particularly for developing early-stage heart disease detection systems and generating predictive machine learning models.
        With its comprehensive feature set, this dataset serves as a valuable resource for advancing heart disease research and improving diagnostic methodologies."""
    },
    "Cardiovascular diseases (Iraq)": {
        "data": pd.read_csv("data/Medicaldataset_final.csv"),
        "description": """The source of the data: https://data.mendeley.com/datasets/wmhctcrt5v/1 \n
        The heart attack dataset was collected at Zheen Hospital in Erbil, Iraq, from January 2019 to May 2019.
        The attributes of this dataset include age, gender, heart rate, systolic blood pressure, diastolic blood pressure, blood sugar, CK-MB, and troponin levels, along with a classification output indicating either a heart attack or no heart attack.
        The gender column is normalized, with males set to 1 and females to 0. The blood sugar (glucose) column is set to 1 if the level is greater than 120, otherwise, it is set to 0. The classification output is set to 1 for positive (heart attack) and 0 for negative (no heart attack)."""
    },
    "Diabetes (America)": {
        "data": pd.read_csv("data/dataset_37_diabetes_final.csv"),
        "description": """The source of the data http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets.\n 
        This dataset includes several hundred rural African-American patients. Patients without a hemoglobin A1c measurement were excluded.
        Those with a hemoglobin A1c of 6.5 or greater were labeled as diabetic (diabetes = yes). Out of 390 patients, 60 were found to be diabetic."""
    },
    "Diabetes (Pima Indians)": {
        "data": pd.read_csv("data/diabetes_classification_final.csv"),
        "description": """The source of the data: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database \n
        This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
        The objective of the dataset is to diagnostically predict whether a patient has diabetes based on certain diagnostic measurements.
        All patients are females at least 21 years old of Pima Indian heritage. The dataset consists of several medical predictor variables and one target variable, Outcome.
        Predictor variables include the number of pregnancies, BMI, insulin level, age, and other relevant medical measurements."""
    }
}

with open("../results.json", 'r') as file:
    performance_metric = json.load(file)

algorithms: Dict[str, BaseEstimator] = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Naive Bayes": GaussianNB,
    "Decision Tree": DecisionTreeClassifier,
    "K nearest neighbours": KNeighborsClassifier
}

@app.route('/')
def index():
    """Render the index page with dataset descriptions."""
    return render_template('index.html', datasets={name: data["description"] for name, data in datasets.items()})

@app.route('/dataset/<dataset_name>', methods=['GET', 'POST'])
def dataset(dataset_name):
    """Render the dataset page and handle column selection and algorithm choice."""
    dataset = datasets[dataset_name]["data"]
    description = datasets[dataset_name]["description"]
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')
        algorithm_name = request.form['algorithm']
        return redirect(url_for('algorithm', dataset_name=dataset_name, algorithm_name=algorithm_name, columns=','.join(selected_columns)))
    return render_template('dataset.html', dataset_name=dataset_name, columns=dataset.columns, algorithms=algorithms, description=description)

@app.route('/algorithm/<dataset_name>/<algorithm_name>/<columns>', methods=['GET', 'POST'])
def algorithm(dataset_name: str, algorithm_name: str, columns: str):
    """Render the algorithm performance page."""

    selected_columns: List[str] = columns.split(',')
    performance: Union[Dict[str, float], str] = get_model_performance(dataset_name, selected_columns, algorithm_name)
    if request.method == 'POST':
        second_algorithm_name = request.form['second_algorithm']
        second_performance = get_model_performance(dataset_name, selected_columns, second_algorithm_name)
        comparison_data = {
            'algorithm_1': algorithm_name,
            'performance_1': performance,
            'algorithm_2': second_algorithm_name,
            'performance_2': second_performance
        }
        return render_template('compare.html', comparison_data=comparison_data, algorithms=algorithms)
    return render_template('algorithm.html', performance=performance, algorithm_name=algorithm_name, columns=columns, algorithms=algorithms.keys())

def get_model_performance(dataset, selected_columns, algorithm_name) -> Union[Dict[str, float], str]:
    """Get the performance of the selected model on the selected dataset and columns."""

    columns_key: str = ', '.join(sorted(selected_columns))
    name_algo: str = get_right_name_algo(algorithm_name)
    name_data: str = get_right_name_data(dataset)
    performance: Union[Dict[str, float], str] = performance_metric[name_algo][name_data].get(columns_key, "No data available")
    return performance

def get_right_name_algo(name: str) -> str:
    """Convert user-friendly algorithm name to the corresponding class name."""

    match name:
        case "Logistic Regression":
            return "LogisticRegression"
        case "Random Forest":
            return "RandomForestClassifier"
        case "Naive Bayes":
            return "GaussianNB"
        case "Decision Tree":
            return "DecisionTreeClassifier"
        case "K nearest neighbours":
            return "KNeighborsClassifier"

def get_right_name_data(name: str) -> str:
    """Convert user-friendly dataset name to the corresponding key in performance metrics."""

    match name:
        case "Cardiovascular diseases (India)":
            return "Cardiovascular_Disease_Dataset"
        case "Cardiovascular diseases (Iraq)":
            return "Medical_Dataset"
        case "Diabetes (America)":
            return "Diabetes_Dataset_37"
        case "Diabetes (Pima Indians)":
            return "Diabetes_Classification"

if __name__ == '__main__':
    app.run(debug=True)
