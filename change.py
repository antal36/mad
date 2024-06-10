import pandas as pd
df = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/heart_attack1/Medicaldataset.csv")
df["Result"] = df["Result"].map({"positive":1, "negative":0})
df.to_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/heart_attack1/Medicaldataset_final.csv", index=False)


df2 = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/dataset_37_diabetes.csv")
df2["'class'"] = df2["'class'"].map({"tested_positive":1, "tested_negative":0})
df2.to_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/dataset_37_diabetes_final.csv", index=False)

df3 = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset.csv")
df3.drop('patientid', axis="columns", inplace=True)
df3["Gender"] = df3["Gender"].map({"male":1, "female":0})
df3.to_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset/Cardiovascular_Disease_Dataset_final.csv", index=False)

df4 = pd.read_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/informatics-edu-diabetes-prediction/informatics-edu-diabetes-prediction/data/diabetes_classification.csv")
df4.drop('patient_number', axis="columns", inplace=True)
df4["diabetes"] = df4["diabetes"].map({"Diabetes":1, "No diabetes":0})
df4.to_csv("C:/Users/antal/Desktop/matfyz/MAD/mad/informatics-edu-diabetes-prediction/informatics-edu-diabetes-prediction/data/diabetes_classification_final.csv", index=False)

"""Original files were deleted after the changes were applied to them, so this code would not run now."""