import numpy as np
import random


def assign_clusters(img_flat, centroids):
    # Fuction implementing clustering assigmnet step of k-means algorithm.
    # It returns the assigned clusters (clusters) and the distances of each pixel to each
    # of the clusters (dist_matrix).
    #
    # Inputs:
    #   img_flat  : Flattened image, numpy array, size of (ncols * nrows, nchannels)
    #   centroids : Centorids (means) of the clusters,numpy array, size of
    #               (number of clusters, nchannels)
    # Outputs:
    #   clusters  : Assigned clusters from 0 to K-1 (K: number of clusters),
    #               size of (ncols * nrows, 1)
    #   dist_matrix : Distance matrix, numpy array, size of (nrows * ncols,
    #                 number of clusters)


    ###################### PLEASE FILL IN HERE ############################
    dist_matrix = np.sqrt(((img_flat - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    clusters = np.argmin(dist_matrix, axis=0)
    return clusters, dist_matrix


def update_centroids(img_flat, clusters, K):
    # Function that implements update step of k-means algorithm.
    # It returns the centroids (means) that are updated.
    #
    # Inputs:
    #   img_flat  : Flattened image, numpy array, size of (ncols * nrows, nchannels)
    #   clusters  : Clusters to which pixels were assigned, size of (ncols *
    #               nrows, 1)
    #
    # Outputs:
    #   centroids : Centorids (means) of the clusters,numpy array, size of
    #               (number of clusters, nchannels)


    ###################### PLEASE FILL IN HERE ############################
    centroids = np.array([img_flat[clusters == k].mean(axis=0) for k in range(K)])
    return centroids


def cluster_with_kmeans(img, K, max_iter=1e4, tol=1e-4):
    # This is the function that takes an input image with one or more channels
    # and runs k-means algorithm on it to cluster its pixels
    # It iteratively calls assign_clusters and update_centroids functions
    # and returns the clsutered image.
    #
    # Inputs:
    #   img     : Image of size (nrows, ncols) or (nrows, ncols, nchannels)
    #             Numpy array
    #   K       : Number of clusters, integer
    #   max_iter: Maximum number of iterations given to k-means algorithm, integer


    # Image size to be checked -- is it a multi-channel image?
    if len(img.shape) < 3:  # One-channel
        nrow, ncol = img.shape
        nchannel = 1
        img_flat = np.reshape(img, (img.size, 1))

    else:  # Multi-channel
        nrow, ncol, nchannel = img.shape
        img_flat = np.reshape(img, (nrow * ncol, nchannel))

    # Initialize the centroids, i.e. means
    init_centroids = img_flat[random.sample(range(nrow * ncol), K), :]
    centroids = init_centroids
    dist_prev = 0

    # Iterate between assignement and updates steps
    for iter in range(max_iter):

        # Assignment step
        clusters, dist_matrix = assign_clusters(img_flat, centroids)

        # Update step
        centroids = update_centroids(img_flat, clusters, K)

        # Do assignments not change much? If so, stop.
        if (abs(np.sum(dist_matrix) - dist_prev) < tol):
            break

        # This is to compare the new cluster assignments with the previous assignment
        dist_prev = np.sum(dist_matrix)

    # Reshape clusters back into the image shape
    img_clustered = np.reshape(clusters, (nrow, ncol))

    print('Converged in {:d} iterations...'.format(iter))
    # Return clustered image
    return img_clustered