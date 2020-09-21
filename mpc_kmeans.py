from sklearn import datasets, metrics
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import time

X, y = datasets.load_iris(return_X_y=True)



mustLink = np.array(([118, 106],[118, 100],[106, 118]))
cantLink = np.array(([38, 118],[38, 106],[38, 100]))

mustLink1 = np.array(([118, 106],[118, 100],[106, 118],[117, 103],[117, 100],[104, 118]))
cantLink1 = np.array(([38, 118],[38, 106],[38, 100],[37, 115],[37, 106],[36, 100]))

mustLink2 = np.array(([118, 106],[118, 100],[106, 118],[117, 103],[117, 100],[104, 118],  [50, 98],[98, 50],[100, 106]))
cantLink2 = np.array(([38, 118],[38, 106],[38, 100],[37, 115],[37, 106],[36, 100],[50, 106],[50, 100],[98, 118]))

constraints= []
rand_index= []
time_total=[]
for i in range(3):
    if(i==0):
        start_time = time.time()
        clusterer = PCKMeans(n_clusters=3)
        clusterer.fit(X, ml=mustLink, cl=cantLink)
        rand_index.append((metrics.mutual_info_score(y, clusterer.labels_)))
        constraints.append((len(mustLink)+len(cantLink)))
        time_total.append((time.time() - start_time))
        
    elif (i==1):
        start_time = time.time()
        clusterer = PCKMeans(n_clusters=3)
        clusterer.fit(X, ml=mustLink1, cl=cantLink1)
        rand_index.append((metrics.mutual_info_score(y, clusterer.labels_)))
        constraints.append((len(mustLink1)+len(cantLink1)))
        time_total.append((time.time() - start_time))
    else:
        start_time = time.time()
        clusterer = PCKMeans(n_clusters=3)
        clusterer.fit(X, ml=mustLink2, cl=cantLink2)
        rand_index.append((metrics.mutual_info_score(y, clusterer.labels_)))
        constraints.append((len(mustLink2)+len(cantLink2)))
        time_total.append((time.time() - start_time))

    plt.title("IRIS PCM Cluster Constraints_set %i" %(i+1))
    plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_, s=50, cmap='viridis')
    centers = clusterer.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


plt.figure()
plt.title('RI vs Constraints')
plt.xlabel('constraints')
plt.ylabel('rand_index')
plt.scatter(constraints,rand_index)

plt.show()

plt.figure()
plt.title('Execution time vs Constraints')
plt.xlabel('constraints')
plt.ylabel('execution time (S)')
plt.scatter(constraints,time_total)
plt.show()
