import numpy as np
from sklearn.cluster import KMeans

def divisive_clustering(X, k):
    clusters = [X]

    while len(clusters) < k:
        largest_cluster = max(clusters, key=len)
        clusters.remove(largest_cluster)

        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(largest_cluster)

        cluster1 = largest_cluster[labels == 0]
        cluster2 = largest_cluster[labels == 1]

        clusters.append(cluster1)
        clusters.append(cluster2)

    return clusters


X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0]])

clusters = divisive_clustering(X,2)

for i,c in enumerate(clusters):
    print("Cluster",i,":",c)