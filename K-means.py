import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from copy import deepcopy



def euclidean_distance(p1, p2):

    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def k_means(data):

    k=3

    data = np.array(list(zip(data["x"], data['y'])))

    #create random K number of centroids

    centroids = np.empty((k, 2))

    for x in range(k):
        centroids[x, 0] = random.randint(np.min(data[:, 0]), np.max(data[:, 0]))
        centroids[x, 1] = random.randint(np.min(data[:, 1]), np.max(data[:, 1]))

    #list of clusters for each data point
    clusters = np.zeros([len(data[:, 0]), 2])
    previous_centroids = np.zeros(centroids.shape)




    while((centroids != previous_centroids).all()):

        previous_centroids = deepcopy(centroids)

        #find centroid for each data point
        for x in range(len(data[:, 0])):
            dist = []

            for j in range(len(centroids[:, 0])):
            
                dist.append(euclidean_distance(data[x], centroids[j]))  
            
            clusters[x, 0] = np.argmin(dist)
            clusters[x, 1] = min(dist)


        #recalculate centroids 
        for x in range(len(centroids[:, 0])):

            try:
                centroids[x, 0] = np.mean(data[np.where(clusters[:, 0] == x), 0])
                centroids[x, 1] = np.mean(data[np.where(clusters[:, 0] == x), 1])
            except Exception:
                pass


    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*")
    plt.show()



    for x in range(len(clusters[:, 0])):

        print("X-Axis - " + str(data[x, 0]))
        print("Y-Axis - " + str(data[x, 1]))
        print("Cluster - " + str(clusters[x, 0]))
        print("------------")


#create data set to be classfied 
data = pd.DataFrame({"x": [1, 2, 2, 1, 3, 6, 7, 6, 7, 6, 2, 7, 8, 6, 7, 3, 8, 8, 8], 
"y":[1, 2, 3, 4, 4, 7, 8, 8, 6, 9, 5, 8, 9, 7, 8, 1, 4, 6, 9]})


model = k_means(data)
