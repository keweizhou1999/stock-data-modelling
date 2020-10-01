# main.py
# Perform k-means clustering and Markov Model

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import random
from statistics import mean
import cluster2 as cluster
import rnn
from sklearn import metrics
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
import optimization as opt
from scipy.optimize import NonlinearConstraint
import tensorflow as tf

###### FUNCTIONS #########

#-----------------------Visualizing Data-------------------------
def draw_vector(v0, v1, ax=None):
    ''' draws eigen vectors
    '''
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', color='b', linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def draw_ellipse(data, cluster):
    '''Plot all of the eclipses, and plot each cluster in different color
    '''
    plt.figure()
    ax = plt.gca()
    for i in range(len(data)):
        if cluster[i] == 0:
            color='r' 
        elif cluster[i] == 1:
            color='g'
        elif cluster[i] == 2:
            color = 'c'
        elif cluster[i] == 3:
            color = 'y'
        elif cluster[i] == 4:
            color = 'k'
        elif cluster[i] == 5:
            color = 'm'
        else:
            color='b'
        L = data[i]
        ellipse = Ellipse(xy=(L[0], L[1]), width=2*L[2], height=2*L[3],angle=L[4], 
                                edgecolor=color, fc='None', lw=2)
        ax.add_patch(ellipse)

        # Uncomment if want to see the scales on both axis are the same
        #
        # plt.xlim(-6000,6000)
        # plt.ylim(-6000,6000)
        plt.margins(1,1)
    plt.show()


#-----------------------Determining K-------------------------
def cluster_distance(clusters):
    """ Sum the average distance in each cluster, and then take the average
    Input: clusters is a array in an array, saving all the eclipses by its cluster
    [[ellipses in cluster 0]
     [ellipses in cluster 1] 
     ... 
     [ellipses in cluster k-1]]
    This array is the output of the cluster.display function
    output: floating point number that is the average distance between all clusters
    """
    clusters_avg = 0
    total = 0.0
    for i in range(len(clusters)):
        c = clusters[i]
        mean = np.mean(c, axis=0)
        for j in range(len(c)):
            total += cluster.distance(c[j], mean)
        clusters_avg += total*1.0/len(c)
    return float(clusters_avg/len(clusters))

def threshold_plot(L):
    """
    Using the output of random_distance, plot distance vs. k for lists of k values
    input: L is a list of paired k values and distances
    """
    # Plot the distances against k
    k = [i[0] for i in L]
    distances = [i[1] for i in L]
    plt.scatter(k, distances)
    plt.xlabel('k')
    plt.ylabel('distance')
    plt.show()

""" Functions Used to Determine Cluster of unused testing data """
def cluster_mean(clusters):
    meanList = []
    for i in range(len(clusters)):
        c = clusters[i]
        mean = np.mean(c, axis = 0)
        # print(mean)
        meanList.append(mean)
    return meanList

def assign_clusters(weight, meanList, dataToBeAssigned):
    dataCluster = []
    for data in dataToBeAssigned:
        distanceList = []
        for clusterMean in meanList:
            distanceList += [cluster.distance(weight, data, clusterMean)]
        dataCluster += [np.argmin(distanceList)]
    return dataCluster

#--------------------------------------Markov Chain---------------

def get_cluster_dict(L, seq_length):
    """ 
    input: a list of cluster labels in time series
    output: an dictionary where the keys are chunks of seq_length clusters 
    and the values is [the most probable fifth cluster, probability of the most probable]
    """
    items = dict()
    for i in range(len(L)-seq_length):
        labels = ''.join(map(str, L[i:i+seq_length] )) 
        if labels not in items:
            items.update({labels: [0,0,0,0,0,0,0]})
        old_value = items.get(labels)
        index = L[i + seq_length]
        old_value[index] += 1
        items.update({labels: old_value})
            
    for key in items:
        max_cluster = np.argmax(items.get(key))
        prob = max(items.get(key))/sum(items.get(key))
        items[key] = [max_cluster, prob]

    return items

def test_markov(dict, testL, seq_length):
    '''
    input: dictionary generated from running get_cluster_dict, testL is the test set, 
            seq_length is the length of the key
    output: the accuracy of prediction and the number of cases in which the testing set's
    shift window was not found in the training set. 
    '''
    not_found = 0
    failed_cases = 0
    for i in range(len(testL)-seq_length-1):
        labels = ''.join(map(str, testL[i:i+seq_length])) 
        if labels not in dict:
            not_found += 1
        elif (dict[labels][0] != testL[i+seq_length]):
            failed_cases += 1
    correctness = 1-((failed_cases + not_found)* 1.0 /(len(testL)))
    return correctness, not_found


def dictionary(Xtrain, Xtest, seq_length):
    """
    Returns the accuracy for a specific sequence length 
    Input: Xtrain, the training data (list)
           Xtest, the testing data (list)
           seq_length, the sequence length of the key (integer)
    """
    # Training and Testing Markov Model
    dictionary = get_cluster_dict(Xtrain, seq_length)
    accuracy, not_found = test_markov(dictionary, Xtest, seq_length)
    # print("Accuracy is " + str(accuracy*100) + "%")
    # print("Cases not found: ", not_found)
    return accuracy

def preprocess_test_data(weight, clusters):
    """
    Pre-processing the testing data
    Input: clusters - an array of time-series label of the training data
    Note: Called in accuracy_vs_length
    """
    # Obtain testData
    dTest  = pd.read_csv("Processed95-00.csv")
    dataframeTest = dTest.loc[:, ['PriceChange', 'VolumeChange']]
    Y = np.array(dataframeTest.to_numpy())
    testData = sum30Day(Y)

    # Get cluster mean
    meanCluster = cluster_mean(clusters)
    # Normalize TestData
    raw_testdata =  np.asarray(testData, dtype=np.float32)
    (norm_testdata, mins, maxs) = cluster.mm_normalize(raw_testdata)
    Xtest = assign_clusters(weight, meanCluster,norm_testdata)
    return Xtest
    

def accuracy_vs_length(weight, clusters, Xtrain):
    print("accuracy vs length")
    """
    Plot the sequence legnth vs accuracy plot
    Input: clusters - a list of time-series label of the training data
    """
    # Pre-processes the test data and return a list of time-series label
    # of the testing data
    Xtest = preprocess_test_data(weight, clusters)

    # generate a list of different sequence lengths
    seq_length_L = [i for i in range(4, 10)]
    accuracy_L = []

    # For each sequence length, determines the accuracy by calling the 
    # dictionary function. Then, append the accuracy to a list
    for seq_length in seq_length_L:
        accuracy = dictionary(Xtrain, Xtest, seq_length)
        accuracy_L.append(accuracy)
    
    # plot sequence legnth vs accuracy
    plt.scatter(seq_length_L, accuracy_L)
    plt.xlabel('seq_length')
    plt.ylabel('accuracy')
    plt.show()


"""Helper Function for Main"""
def sum30Day(dataframe):
    '''
    Summarizing 30 days data in the following format:
        [mean of percent price change, mean of percent volume change change,
        principal eigenvalue, secondary principal eigenvalue, theta]
    '''
    # data is the matrix that holds all the pca 5-elements lists
    # it has a dimension of (n, 5) where n is the number of pcas we have
    data = []

    # Perform PCA on every 30 data points using the shifting strategy
    i = 0
    pca = PCA() # declare PCA object with constructor

    while (i<len(dataframe)-30):
        oneMonth = dataframe[i:i+30]
        returnList=[]

        # Append mean of % change price and % change volume
        returnList.append(mean(oneMonth[:, 0]))
        returnList.append(mean(oneMonth[:, 1]))
        
        # Append the two eigenvalues, the bigger eigenvalues go first
        # and hold more weight
        pca.fit(oneMonth)
        evals = pca.explained_variance_
        returnList.append(evals[0])
        returnList.append(evals[1])

        # calculate theta
        y = pca.components_[0][1]
        x = pca.components_[0][0]
        theta = math.atan(y/x)*180/(math.pi)

        # append theta
        returnList.append(theta)

        # increment i 
        i = i+1

        # append returnList to data
        data.append(returnList)

    return data 

###### MAIN ########

def main():
    #--------Read in old variables data-------
    # read in preprocessed values
    d  = pd.read_csv("newProcessed.csv")
    dataframe = d.loc[:, ['PriceChange', 'VolumeChange']]
    # designated weight
    weight = [0.2, 0.78, 0.015, 0.005]

    #--------Read in new variables data--------
    # # Uncomment this to run on new variable data
    # #read in preprocessed values
    # d  = pd.read_csv("SandPMarch31.csv")
    # dataframe = d.loc[:, ['PriceChange', 'frac']]
    # #designated weight
    # weight = [0.09, 0.1, 0.1, 0.71]

    #--------Pre-process data------------------
    X = np.array(dataframe.to_numpy())
    #summarize 30 day data and put into matrix
    data = sum30Day(X)
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(2)
    
    # convert data to np.array
    raw_data =  np.asarray(data, dtype=np.float32)

    # define the number of clusters 
    k = 7

    #---------Optimization of parameters--------------
    ### Uncomment to either manually or use scipy to optimize weight
    # # get the sihouette score
    # def rosen(weight):
    #     return opt.opt_helper(k, weight, raw_data)

    # # manually test different weights
    # opt.manual_minimize(rosen)

   
    # # calculate distance depending on the weight 
    # def distance(item1, item2):
    #     return cluster.distance(weight,item1, item2)
    # # use the scipy minimization to find optimal parameters
    # opt.minimizeHelper(rosen, weight)
    
    #-------------K-mean cluster once-----------
    # Perform K-mean cluster once for the designated weight, in order
    # to generate the Markov Chain plot
    print("\nBegin k-means clustering demo \n")
    # normalize the raw data so that they are all in the range of (0,1)
    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)

    # perform clustering
    print("\nClustering normalized data with k=" + str(k))
    clustering = cluster.cluster(weight, norm_data, k)
    print("\nDone. Clustering:")

    print("\nRaw data grouped by cluster: ")
    clusters = cluster.display(norm_data, clustering, k)
    
    #---------------Markov Chain Plot-------------------
    # # Setting the trained data to Xtrain
    # Xtrain = clustering
    # # create the accuracy vs sequence length plot
    # accuracy_vs_length(weight, clusters, Xtrain)

    #-------------Use average distance to find optimal K-----------
    # # Uncoment if want to visualize the optimal k value, must comment out the block above
    # # Find the optimal k value by calculating the average distance associated with each
    #
    # distance_L = []
    # for  k in range(1, 8):
    #     print("k = "+ str(k))
    #     clustering = cluster.cluster(norm_data, k)
    #     clusters = cluster.display(norm_data, clustering, k)
    #     distance = cluster_distance(clusters)
    #     distance_L.append([k, distance])
    # threshold_plot(distance_L)

    #------------Create RNN prediction -----------------------
    # # Uncommet to perform RNN Prediction
    # accuracy_L = []
    # for seq_length in range(10,25):
    #     rnn.rnn(clustering, seq_length, k)
    #     checkpoint_dir = './training_checkpoints_' + str(seq_length)
    #     # Restore the latest checkpoint
    #     tf.train.latest_checkpoint(checkpoint_dir)
    #     embedding_dim = 256
    #     rnn_units = 1024
    #     model = rnn.build_model(k, embedding_dim, rnn_units, batch_size=1)
    #     model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    #     model.build(tf.TensorShape([1, None]))

    #     predictions_L = rnn.predict(model, clustering, seq_length)
    #     actual_L = clustering[seq_length:]
    #     accuracy = rnn.accuracy(predictions_L, actual_L)
    #     accuracy_L.append(accuracy)
    #     print("accuracy of seq_length ", seq_length, " is ", str(accuracy))
    # print(accuracy_L)

   
if __name__ == "__main__":
    main()
