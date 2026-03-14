import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = np.array([[1,2],[2,2],[2,3],
              [8,7],[8,8],[25,80]])

model = DBSCAN(eps=3, min_samples=2)

labels = model.fit_predict(X)

print("Cluster labels:", labels)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()