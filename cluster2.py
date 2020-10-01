# cluster.py
# Used for performing k-means clustering with custom distance function
# Imported to main.py

import numpy as np
import math as math

def mm_normalize(data):
    """
    normalize all values by doing (x-min)/ (max - min) to get all values
    between 0 and 1
    input: a numpy array of raw data
    output: a list containing the normalized version of numpy's array of raw data,
    and lists of mins and max
    """
    (rows, cols) = data.shape  
    mins = np.zeros(shape=(cols), dtype=np.float32)
    maxs = np.zeros(shape=(cols), dtype=np.float32)
    for j in range(cols):
        mins[j] = np.min(data[:,j])
        maxs[j] = np.max(data[:,j])

    result = np.copy(data)
    for i in range(rows):
        for j in range(cols):
            result[i,j] = (data[i,j] - mins[j]) / (maxs[j] - mins[j])
    return (result, mins, maxs)

def distance(weight, item, mean):
    '''
    item: a list of 5 items: coordinates of center, the principal and
    secondary eigenvalue and theta
    mean: a vector containing these 5 components to all of our data
    weight: a list of 4 items that contained the weight of the center, 
    the principal and the secondary eigenvalue, and theta
    '''
    # Weight for center
    CenterW = weight[0] 
    # Weight for Prinipal Eigenvalue     
    PrincipalW = weight[1]   
    # Weight for the smaller Eigenvalue 
    MinorW = weight[2]
    AngleW = weight[3]
        
    # Putting the coordinates of center into an list
    center1 = item[0:2]
    center2 = mean[0:2]

    # Putting the coordinates of eigenvalues into an list
    evalue1 = item[2:4]
    evalue2 = mean[2:4]

    theta1 = item[-1]
    theta2 = mean[-1]

    center_dis = np.linalg.norm(center2 - center1)
    evalue_dis = abs(evalue2[0]-evalue1[0])
    evalue_dis2 = abs(evalue2[1]-evalue1[1])
    theta_dis = abs(theta2-theta1)
    return CenterW * center_dis**2 + PrincipalW * evalue_dis**2 + MinorW * evalue_dis2**2 + AngleW * theta_dis**2

def update_clustering(weight, norm_data, clustering, means):
    """
    given a new set of means, assign new clustering
    return False if no change or bad clustering
    weight: a list of 4 items that contained the weight of the center, 
    the principal and the secondary eigenvalue, and theta
    """
    n = len(norm_data)
    k = len(means)

    new_clustering = np.copy(clustering)  # proposed new clustering
    distances = np.zeros(shape=(k), dtype=np.float32)  # from item to each mean

    for i in range(n):  # go thru each data item
        for kk in range(k):
            distances[kk] = distance(weight, norm_data[i], means[kk])  
        new_id = np.argmin(distances)
        new_clustering[i] = new_id
    
    if np.array_equal(clustering, new_clustering):  # no change, then clustering is complete
        return False

    # make sure that no cluster counts have gone to zero
    counts = np.zeros(shape=(k), dtype=np.int)
    for i in range(n):
        c_id = clustering[i]
        counts[c_id] += 1
    
    for kk in range(k): 
        if counts[kk] == 0:  # bad clustering if cluster counts gone to 0
            return False

    # there was a change, and no counts have gone 0
    for i in range(n):
        clustering[i] = new_clustering[i]  # update the label for cluster
    return True

def update_means(norm_data, clustering, means):
    """
    given a new clustering, compute new means
    assumes update_clustering has just been called
    to guarantee no 0-count clusters
    """
    (n, dim) = norm_data.shape
    k = len(means)
    counts = np.zeros(shape=(k), dtype=np.int)
    new_means = np.zeros(shape=means.shape, dtype=np.float32)  # k x dim
    for i in range(n):  # walk thru each data item
        c_id = clustering[i]
        counts[c_id] += 1
        for j in range(dim):
            new_means[c_id,j] += norm_data[i,j]  # accumulate sum

    for kk in range(k):  # each mean
        for j in range(dim):
            new_means[kk,j] /= counts[kk]  # assumes not zero

    for kk in range(k):  # each mean
        for j in range(dim):
            means[kk,j] = new_means[kk,j]  # update by ref

def initialize(norm_data, k):
    (n, dim) = norm_data.shape
    clustering = np.zeros(shape=(n), dtype=np.int)  # index = item, val = cluster ID
    for i in range(k):
        clustering[i] = i
    for i in range(k, n):
        clustering[i] = np.random.randint(0, k) 

    means = np.zeros(shape=(k,dim), dtype=np.float32)
    update_means(norm_data, clustering, means)
    return(clustering, means) 

def cluster(weight, norm_data, k):
    """
    Perform k-means clustering by calling update_clustering and
    update_means
    """
    (clustering, means) = initialize(norm_data, k)

    ok = True  # if a change was made and no bad clustering
    max_iter = 100
    sanity_ct = 1
    while sanity_ct <= max_iter:
        ok = update_clustering(weight, norm_data, clustering, means)  # use new means
        if ok == False:
            break  # done
        update_means(norm_data, clustering, means)  # use new clustering
        sanity_ct += 1

    return clustering

def display(raw_data, clustering, k):
    """
    display all input data according to their cluster, 
    prints out label of the cluster
    """
    (n, dim) = raw_data.shape
    # print("-------------------")
    clusters = []
    for kk in range(k):  # group by cluster ID
        kth_cluster = []
        for i in range(n):  # scan the raw data
            c_id = clustering[i]  # cluster ID of curr item
            if c_id == kk:  # curr item belongs to curr cluster so . . 
                kth_cluster.append(raw_data[i])
                print("%4d " % i, end=""); print(raw_data[i])
        print("-------------------")  
        clusters.append(kth_cluster)
    return clusters


