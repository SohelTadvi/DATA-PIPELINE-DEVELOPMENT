import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Data
file_path = r'C:\Users\Owner\Documents\Python Panda\traffic_accidents.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Handle possible encoding errors during file reading
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='latin1')

# Check if the dataset is loaded correctly
if data.empty:
    raise ValueError("Dataset is empty. Please check the file.")

print("Dataset loaded successfully!")
print("Columns in dataset:", data.columns)

# Step 2: Identify Features and Target Variable
target_column = 'crash_type'  # Replace with actual target column
# Ensure the target column exists in the dataset
if target_column not in data.columns:
    raise KeyError(f"Target column '{target_column}' not found in the dataset.")

# Separate features (X) and target (y)
features = data.drop(target_column, axis=1)
target = data[target_column]

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Identify Column Types
# Identify numerical and categorical columns in the features
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# **Fix 1: Remove High Cardinality Columns (like dates)**
# Identify columns with high cardinality (i.e., columns with a large number of unique values, like dates)
high_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() > 100]
# Drop high-cardinality columns from both training and test datasets
X_train = X_train.drop(columns=high_cardinality_cols, errors='ignore')
X_test = X_test.drop(columns=high_cardinality_cols, errors='ignore')
# Update the categorical columns list after removing high cardinality columns
categorical_cols = list(set(categorical_cols) - set(high_cardinality_cols))  # Update categorical cols

# Print the column types after processing
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)
print("Removed high-cardinality columns:", high_cardinality_cols)

# **Fix 2: Change OneHotEncoder to sparse=True**
# Create separate transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler())  # Standardize numerical features to zero mean and unit variance
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))  # One-hot encode categorical features (use sparse matrices for efficiency)
])

# Combine numerical and categorical transformations in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),  # Apply numerical transformer to numerical columns
        ('cat', categorical_transformer, categorical_cols)  # Apply categorical transformer to categorical columns
    ])

# **Fix 3: Reduce Model Complexity (Fewer Trees)**
# Create a machine learning pipeline with a preprocessor and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing to the data
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42))  # Use a RandomForestClassifier with fewer trees and max depth for model simplicity
])

# Step 5: Train the Model
pipeline.fit(X_train, y_train)  # Fit the model to the training data

# Step 6: Evaluate the Model
y_pred = pipeline.predict(X_test)  # Predict on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
print(f'Model accuracy: {accuracy:.2f}')  # Print the accuracy of the model
