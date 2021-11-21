import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df = pd.read_csv("Star_with_gravity.csv")
x = df.iloc[:,[3,4]].values

wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42)
  kmeans.fit(x)
  wcss.append((kmeans.inertia_))

plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel('Number of clusters')
plt.show()
print(x)