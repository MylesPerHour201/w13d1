import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

dataset = pd.read_csv('employee_attrition.csv')

print(dataset.head())

print(dataset.shape)

target = 'Attrition'

features = dataset.columns[dataset.columns != target]

missing_values = dataset.isnull().sum()
if missing_values.sum() == 0:
    print('There are no missing values')
else:
    clean_data = dataset.dropna()
    print(f'This is the clean data {clean_data.isnull().sum()}')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# Perform one-hot encoding for categorical features
categorical_features = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(dataset_imputed[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
dataset_processed = pd.concat([dataset_imputed.drop(columns=categorical_features), encoded_df], axis=1)

# Split the dataset into training and testing sets
X = dataset_processed.drop(columns=[target])
y = dataset_processed[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



import numpy as np

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")
