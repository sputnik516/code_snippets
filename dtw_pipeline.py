import pandas as pd
import numpy as np
from dtw import accelerated_dtw
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load your dataset and dictionary
data = {
    'sequence_1': [1.0, 2.0, 3.0, 4.0, 5.0],
    'sequence_2': [2.0, 3.0, 4.0, 5.0, 6.0],
    'sequence_3': [1.5, 2.5, 3.5, 4.5, 5.5],
    'sequence_4': [2.5, 3.5, 4.5, 5.5, 6.5],
}
df = pd.DataFrame(data)

dict_values = {
    'sequence_1': 10,
    'sequence_2': 20,
    'sequence_3': 15,
    'sequence_4': 25,
}

# Step 1: Perform DTW analysis and clustering
num_sequences = len(df.columns)
dtw_matrix = np.zeros((num_sequences, num_sequences))

for i in range(num_sequences):
    for j in range(i + 1, num_sequences):
        seq1, seq2 = df.iloc[:, i].values, df.iloc[:, j].values
        dtw_distance, _, _, _ = accelerated_dtw(seq1, seq2, dist='euclidean')
        dtw_matrix[i, j] = dtw_distance
        dtw_matrix[j, i] = dtw_distance

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
pattern_clusters = kmeans.fit_predict(dtw_matrix)

# Step 2: Fit a clustering model on the dictionary values, dataset columns, and pattern clusters
features = []
labels = []

for sequence_name, cluster in zip(df.columns, pattern_clusters):
    features.append([dict_values[sequence_name]] + list(df[sequence_name].iloc[:10].values))
    labels.append(cluster)

X_train, _, y_train, _ = train_test_split(features, labels, train_size=0.8, random_state=42)

# Standardize the features and use K-nearest neighbors classifier for prediction
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), list(range(len(X_train[0]))))])
knn = KNeighborsClassifier(n_neighbors=3)
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', knn)])
pipe.fit(X_train, y_train)

# Step 3: Predict the pattern clusters for sequences in the testing dataset
test_data = {
    'sequence_5': [1.0, 2.0, 3.0, 4.0, 5.0],
    'sequence_6': [2.0, 3.0, 4.0, 5.0, 6.0],
}

test_dict_values = {
    'sequence_5': 12,
    'sequence_6': 22,
}

test_features = []

for sequence_name in test_data:
    test_features.append([test_dict_values[sequence_name]] + list(test_data[sequence_name][:10]))

predicted_clusters = pipe.predict(test_features)

# Print the predicted clusters
for sequence_name, cluster in zip(test_data.keys(), predicted_clusters):
    print(f"{sequence_name} belongs to cluster {cluster}")
