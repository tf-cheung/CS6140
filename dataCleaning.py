import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the data
data = pd.read_csv('train.csv', low_memory=False)

# Remove columns with more than 70% missing values
threshold = 0.7 * len(data)
data = data.dropna(thresh=threshold, axis=1)

# Separate numerical, categorical, and binary features
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns
binary_features = [col for col in numerical_features if data[col].nunique() == 2]

# Update numerical features list, excluding binary features
numerical_features = [col for col in numerical_features if col not in binary_features]
# Remove constant features
numerical_features = [feature for feature in numerical_features if data[feature].std() != 0]

# Impute categorical features
cat_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
data[categorical_features] = cat_imputer.fit_transform(data[categorical_features])

# Impute numerical features
num_imputer = SimpleImputer(strategy='median')
data[numerical_features] = num_imputer.fit_transform(data[numerical_features])

# Impute binary features
binary_imputer = SimpleImputer(strategy='most_frequent')
# Apply the imputer to the binary features in the dataframe
data[binary_features] = binary_imputer.fit_transform(data[binary_features])

# Find unique values in each binary feature
unique_values_in_binary_features = {}
for feature in binary_features:
    unique_values = data[feature].unique()
    unique_values_in_binary_features[feature] = unique_values

# Frequency encoding for categorical features
for col in categorical_features:
    counts = data[col].value_counts()
    data[col] = data[col].map(counts)

# Save the processed data to a file
processed_file_path = 'processed_data.csv'
data.to_csv(processed_file_path, index=False)
