import numpy as np
import random
from scipy.spatial.distance import euclidean


class KMeans():

    def __init__(self):
        """ Constructor
        """
        self.centers = None
        self.n_clusters = 1

    def sorted_kmeans_fit(self, X, n_clusters):
        """ Partition data into clusters. This implementation uses
        kmeans++ initialization which selects initial cluster centers in a way
        that speeds up convergence.
        """
        n_samples = np.shape(X)[0]
        X_mean = np.mean(X, axis=0)  # subtract mean for more accurate distances
        X -= X_mean
        clusters = []

        centers = self.kmeans_plusplus(X, n_clusters)  # initialize centers
        centers, assignments = \
            self.em(X, centers, n_clusters)  # iteratively refine clusters

        sizes = np.zeros((n_clusters))
        # sort clusters based on number of points
        for i in range(len(centers)):
            sizes[i] = len(assignments[i])
        indices = np.flip(np.argsort(sizes))
        new_centers = centers[indices]

        X += X_mean
        centers += X_mean
        self.centers = centers
        self.n_clusters = n_clusters

        return centers

    def cluster_points(self, X, centers, n_clusters):
        """ Assign points to clusters based on Euclidean distance from the point
        to the cluster center.
        """
        clusters = [[] for _ in range(n_clusters)]
        for x in X:
            d, best_center = self.nearest_distance(x, centers)
            try:
                clusters[best_center].append(x)
            except KeyError:
                clusters[best_center] = [x]
        return clusters

    def reevaluate_centers(self, clusters, n_clusters, n_features):
        """ Compute the cluster center.
        """
        new_centers = np.zeros((n_clusters, n_features))
        for i in range(n_clusters):
            cluster = clusters[i]
            size = len(cluster)
            for j in range(size):
                totals = np.zeros((n_features))
                for k in range(n_features):
                    totals[k] += cluster[j][k]
            for k in range(n_features):
                if size == 0:
                    new_centers[i][k] = 0.0
                else:
                    new_centers[i][k] = totals[k] / float(size)
        return new_centers

    def has_converged(self, centers, old_centers, inum):
        """ Determine if kmeans has converged. This is True if the clusters have not
        changed since the previous iteration or if a specified number of iterations
        has been reached.
        """
        max_iter = 300
        if inum == 0:
            return False
        elif np.array_equal(centers, old_centers):
            return True
        elif inum >= max_iter:
            return True
        else:
            return False

    def em(self, X, centers, n_clusters):
        """ Create clusters.
        """
        n_samples, n_features = np.shape(X)
        inum = 0
        oldcenters = centers
        while not self.has_converged(centers, oldcenters, inum):
            inum += 1
            oldcenters = centers
            # assign all points in X to clusters
            clusters = self.cluster_points(X, centers, n_clusters)
            # reevaluate centers
            centers = self.reevaluate_centers(clusters, n_clusters, n_features)
        return (centers, clusters)

    def sorted_kmeans_predict(self, X, centers):
        """ Predict the closest cluster for each sample.
        """
        n_samples = np.shape(X)[0]
        results = np.zeros((n_samples), dtype=np.int_)
        for i in range(n_samples):
            distance, index = self.nearest_distance(X[i], centers)
            results[i] = int(index)
        return results

    def nearest_distance(self, x, centers):
        """ Compute the cluster whose center is closest to the input point.
        Return the distance and the cluster index.
        """
        n_clusters = len(centers)
        distances = np.zeros((n_clusters))
        for i in range(n_clusters):
            distances[i] = euclidean(x, centers[i])
        min_pos = distances.argmin()
        min_val = distances[min_pos]
        return min_val, min_pos

    def kmeans_plusplus(self, X, n_clusters):
        """ Select initial clusters centers using kmeans++.
        """
        n_samples, n_features = np.shape(X)

        centers = np.empty((n_clusters, n_features), dtype=X.dtype)
        indices = np.zeros((n_samples))
        n_local_trials = 2 + int(np.log(n_clusters))  # number of seeding trials

        center_id = np.random.randint(n_samples)  # pick first center randomly

        indices = np.full(n_samples, -1, dtype=int)
        centers[0] = X[center_id]
        indices[0] = 1

        # create list of squared distances between points and closest centers
        closest_dist_sq = np.zeros((n_samples))
        for i in range(len(X)):
            ds, di = self.nearest_distance(X[i], centers)
            ds *= ds
            closest_dist_sq[i] = ds
        total = closest_dist_sq.sum()

        # pick remaining n_clusters-1 points by sampling with probability
        # proportional to squared distance to closest center
        for c in range(1, n_clusters):
            chosen_threshold = np.random.uniform(total)
            cum = 0
            found = False
            for i in range(n_samples):
                cum += closest_dist_sq[i]
                if total < chosen_threshold and indices[i] != 1:
                    centers[c] = X[c]
                    indices[i] = 1
                    found = True
                    break
            if found == False:
                chosen_point = np.random.randint(n_samples)
                centers[c] = X[c]
                indices[chosen_point] = 1
            # create list of squared distances between points and closest centers
            closest_dist_sq = np.zeros((n_samples))
            for i in range(len(X)):
                ds, di = self.nearest_distance(X[i], centers)
                ds *= ds
                closest_dist_sq[i] = ds
            total = closest_dist_sq.sum()

            return centers
