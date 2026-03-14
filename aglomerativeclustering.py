import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

X = np.array([[1,2],[1,4],[1,0],
              [10,2],[10,4],[10,0]])

model = AgglomerativeClustering(n_clusters=2)

labels = model.fit_predict(X)

print("Cluster labels:", labels)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()