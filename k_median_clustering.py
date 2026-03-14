import numpy as np

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def k_median(X, k, max_iter=100):
    centroids = X[np.random.choice(len(X), k)]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]

        for point in X:
            distances = [manhattan_distance(point, c) for c in centroids]
            cluster = np.argmin(distances)
            clusters[cluster].append(point)

        new_centroids = []
        for cluster in clusters:
            cluster = np.array(cluster)
            new_centroids.append(np.median(cluster, axis=0))

        new_centroids = np.array(new_centroids)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters


X = np.array([[1,2],[1,4],[1,0],[10,2],[10,4],[10,0]])

centroids, clusters = k_median(X,2)

print("Centroids:",centroids)