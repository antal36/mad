# import json as js
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier


# df1 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/Cardiovascular_Disease_Dataset_final.csv")
# df2 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/dataset_37_diabetes_final.csv")
# df3 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/diabetes_classification_final.csv")
# df4 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/Medicaldataset_final.csv")

# algorithms = [LogisticRegression, RandomForestClassifier, GaussianNB, DecisionTreeClassifier, KNeighborsClassifier]


# result = {}
# result["df1"] = {}
# result["df1"]["LogisticRegression"] = {}




# # X,y = df1.iloc[:,:-1],df1.iloc[:,-1]

# # clf = LogisticRegression(random_state=0,max_iter=10000).fit(X,y)
# # print(clf.score(X,y))

# X, y = df1.iloc[:, :-1], df1.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Logistic Regression
# logistic_model = LogisticRegression(max_iter=10000)  # max_iter set to ensure convergence
# logistic_model.fit(X_train, y_train)
# y_pred = logistic_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# result["df1"]["LogisticRegression"]["all"]  = accuracy

# # Print the result
# print("Logistic Regression Accuracy:", accuracy)
# print(result)

# json_object = js.dumps(result, indent= 4)

# with open("test.json", "w") as outfile:
#     outfile.write(json_object)

import json as js
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load datasets
df1 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/Cardiovascular_Disease_Dataset_final.csv")
df2 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/dataset_37_diabetes_final.csv")
df3 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/diabetes_classification_final.csv")
df4 = pd.read_csv("C:/Users/varga/OneDrive/Management Dat/projekt/mad/project/data/Medicaldataset_final.csv")

# Define datasets
datasets = {
    "Cardiovascular_Disease_Dataset": df1,
    "Diabetes_Dataset_37": df2,
    "Diabetes_Classification": df3,
    "Medical_Dataset": df4
}

# Define algorithms
algorithms = [LogisticRegression, RandomForestClassifier, GaussianNB, DecisionTreeClassifier, KNeighborsClassifier]

# Initialize result dictionary
result = {}

# Define a function to evaluate a given algorithm on all datasets
def evaluate_algorithm(algorithm):
    algorithm_name = algorithm.__name__
    result[algorithm_name] = {}
    
    for dataset_name, df in datasets.items():
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = algorithm(max_iter=10000) if algorithm == LogisticRegression else algorithm()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        result[algorithm_name][dataset_name] = accuracy

# Evaluate all algorithms on all datasets
for algorithm in algorithms:
    evaluate_algorithm(algorithm)

# Print the results
print(result)

# Dump the results into a JSON file
json_object = js.dumps(result, indent=4)
with open("results.json", "w") as outfile:
    outfile.write(json_object)
