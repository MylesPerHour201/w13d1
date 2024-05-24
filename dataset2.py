import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
dataset = pd.read_csv('employee_attrition.csv')

# Check for missing values
missing_values = dataset.isnull().sum()
if missing_values.sum() == 0:
    print('There are no missing values')
else:
    # Handle missing values by imputing with the most frequent value
    imputer = SimpleImputer(strategy='most_frequent')
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    print('Missing values handled')

# Define target and feature variables
target = 'Attrition'
features = dataset.columns[dataset.columns != target]

# Identify categorical columns for encoding
categorical_features = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
ordinal_features = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']

# Encode ordinal features using LabelEncoder
label_encoders = {}
for feature in ordinal_features:
    le = LabelEncoder()
    dataset[feature] = le.fit_transform(dataset[feature])
    label_encoders[feature] = le

# Perform one-hot encoding for categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(dataset[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Replace original categorical columns with encoded columns
dataset_encoded = pd.concat([dataset.drop(columns=categorical_features), encoded_df], axis=1)

# Split the dataset into training and testing sets
X = dataset_encoded.drop(columns=[target])
y = dataset_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Yes')
recall = recall_score(y_test, y_pred, pos_label='Yes')
f1 = f1_score(y_test, y_pred, pos_label='Yes')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Feature importance
importances = model.feature_importances_
indices = importances.argsort()[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")
