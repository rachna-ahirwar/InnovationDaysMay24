from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from numpy import unique
from numpy import where
from matplotlib import pyplot

# TODO: clustering algorithm - affinity propagation

def clustering_algorithm(filename):
    # # Load your CSV data
    data = pd.read_csv(filename)

    # Convert alphanumeric strings to categorical values and then encode them as integers
    for column in data.columns:
        if data[column].dtype == 'object':
            # Use pandas factorize to convert strings to unique integer codes
            data[column] = pd.factorize(data[column])[0]

    # # Assuming your data has features in all columns
    X = data.values
    # # Scale the features
    scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)

    # # Initialize and fit Affinity Propagation model
    affinity_propagation = AffinityPropagation(damping=0.99, max_iter=1000)
    affinity_propagation.fit(X)

    # # Get cluster labels
    cluster_labels = affinity_propagation.labels_
    print (cluster_labels)
    print (len(unique(cluster_labels)))

    # # Add cluster labels to your original dataframe
    data['Cluster'] = cluster_labels

    # Now data contains your original data with an additional column 'Cluster' indicating the cluster each data point belongs to
    #print(data.head())
    
    # plot the clusters
    for cluster in unique(cluster_labels):
        # get data points that fall in this cluster
        index = where(cluster_labels == cluster)
        # make the plot
        #print (X[index, 0])
        pyplot.scatter(X[index, 0], X[index, 1])

    # show the Gaussian Mixture plot
    pyplot.show()





# # GAUSSIAN MIXTURE
# from sklearn.mixture import GaussianMixture
# from numpy import unique
# from numpy import where
# from matplotlib import pyplot

# def gaussianMixture(filename) :
#     # # Load your CSV data
#     data = pd.read_csv(filename)

#     # Convert alphanumeric strings to categorical values and then encode them as integers
#     for column in data.columns:
#         if data[column].dtype == 'object':
#             # Use pandas factorize to convert strings to unique integer codes
#             data[column] = pd.factorize(data[column])[0]

#     # # Assuming your data has features in all columns
#     X = data.values
#     training_data = X

#     # define the model
#     gaussian_model = GaussianMixture(n_components=5)

#     # train the model
#     gaussian_model.fit(training_data)

#     # assign each data point to a cluster
#     gaussian_result = gaussian_model.predict(training_data)

#     # get all of the unique clusters
#     gaussian_clusters = unique(gaussian_result)

#     # print similar arrays
#     row_no = 20
#     gaussian_cluster = gaussian_model.predict(training_data[row_no].reshape(1, -1))
#     index = where(gaussian_result == gaussian_cluster)
#     out = (training_data[index,0])
#     print (out)
    
#     # plot Gaussian Mixture the clusters
#     for gaussian_cluster in gaussian_clusters:
#         # get data points that fall in this cluster
#         index = where(gaussian_result == gaussian_cluster)
#         # make the plot
#         print (training_data[index, 0])
#         pyplot.scatter(training_data[index, 0], training_data[index, 1])

#     # show the Gaussian Mixture plot
#     pyplot.show()

