import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# sample data
X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0]])

# model
kmeans = KMeans(n_clusters=2, random_state=0)

# fit model
kmeans.fit(X)

# predictions
labels = kmeans.predict(X)

# centroids
centroids = kmeans.cluster_centers_

print("Labels:", labels)
print("Centroids:", centroids)

# visualization
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], c='red', marker='x')
plt.show()
