from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# TODO: clustering algorithm - affinity propagation

def clustering_algorithm(data):
    return data

import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

# # Load your CSV data
data = pd.read_csv('file1.csv')

# # Assuming your data has features in all columns
X = data.values
# # Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Initialize and fit Affinity Propagation model
affinity_propagation = AffinityPropagation()
affinity_propagation.fit(X_scaled)

# # Get cluster labels
cluster_labels = affinity_propagation.labels_

# # Add cluster labels to your original dataframe
data['Cluster'] = cluster_labels

# # Now data contains your original data with an additional column 'Cluster' indicating the cluster each data point belongs to
print(data.head())
