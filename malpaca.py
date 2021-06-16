#!/usr/bin/python3
import pickle
import random
import sys, dpkt, datetime, glob, os, csv
import socket
import seaborn as sns

from models import ConnectionKey
from models import PackageInfo
from speedup import _dtw_distance, cosine_distance
from tqdm import tqdm
from numba import jit

import matplotlib
import matplotlib.colors as colors
from collections import deque, defaultdict

from fastdtw import fastdtw

from scipy.spatial.distance import cdist, pdist, cosine, euclidean, cityblock
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import silhouette_score
from sklearn.cluster._agglomerative import AgglomerativeClustering
from sklearn_extra.cluster._k_medoids import KMedoids
from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.utils.validation import check_symmetric

import seaborn as sns
import hdbscan
import time

# Constants
thresh = 20
random.seed(50)
totalConns = None
expname = 'exp'
if len(sys.argv) > 3:
    expname = sys.argv[3]

if len(sys.argv) > 4:
    thresh = int(sys.argv[4])


def difference(str1, str2):
    return sum([str1[x] != str2[x] for x in range(len(str1))])


def distanceMatrixBytes(data, values):
    matrix = np.zeros((len(data.values()), len(data.values())))
    for a in tqdm(range(len(data.values()))):
        i = np.array([x.bytes for x in values[a]])
        for b in range(len(data.values())):
            j = np.array([x.bytes for x in values[b]])
            if len(i) == 0 or len(j) == 0: continue
            if a == b:
                matrix[a][b] = 0.0
            elif b > a:
                dist = _dtw_distance(i, j)
                matrix[a][b] = dist
                matrix[b][a] = dist
    return matrix


def distanceMatrixGaps(data, values):
    matrix = np.zeros((len(data.values()), len(data.values())))
    for a in tqdm(range(len(data.values()))):
        i = np.array([x.gap for x in values[a]])
        for b in range(len(data.values())):
            j = np.array([x.gap for x in values[b]])
            if len(i) == 0 or len(j) == 0: continue
            if a == b:
                matrix[a][b] = 0.0
            elif b > a:
                dist = _dtw_distance(i, j)
                matrix[a][b] = dist
                matrix[b][a] = dist
    return matrix


def distanceMatrixSource(data, values):
    matrix = np.zeros((len(data.values()), len(data.values())))
    ngrams = []
    for a in range(len(values)):
        profile = dict()
        dat = np.array([x.sourcePort for x in values[a]])
        # ngrams.append(zip(dat, dat[1:], dat[2:]))
        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            if b not in profile.keys():
                profile[b] = 0
            profile[b] += 1
        ngrams.append(profile)
    assert len(ngrams) == len(values)

    for a in tqdm(range(len(ngrams))):
        for b in range(len(ngrams)):

            i = ngrams[a]
            j = ngrams[b]
            ngram_all = list(set(i.keys()) | set(j.keys()))
            i_vec = np.array([(i[item] if item in i.keys() else 0) for item in ngram_all])
            j_vec = np.array([(j[item] if item in j.keys() else 0) for item in ngram_all])

            if a == b:
                matrix[a][b] = 0.0
            elif b > a:

                dist = cosine_distance(i_vec, j_vec)
                matrix[a][b] = dist
                matrix[b][a] = dist
    return matrix


def distanceMatrixDest(data, values):
    matrix = np.zeros((len(data.values()), len(data.values())))
    ngrams = []
    for a in range(len(values)):
        profile = dict()
        dat = np.array([x.destinationPort for x in values[a]])
        # ngrams.append(zip(dat, dat[1:], dat[2:]))
        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            if b not in profile.keys():
                profile[b] = 0
            profile[b] += 1
        ngrams.append(profile)
    assert len(ngrams) == len(values)

    for a in tqdm(range(len(ngrams))):
        for b in range(len(ngrams)):

            i = ngrams[a]
            j = ngrams[b]
            ngram_all = list(set(i.keys()) | set(j.keys()))
            i_vec = np.array([(i[item] if item in i.keys() else 0) for item in ngram_all])
            j_vec = np.array([(j[item] if item in j.keys() else 0) for item in ngram_all])

            if a == b:
                matrix[a][b] = 0.0
            elif b > a:

                dist = cosine_distance(i_vec, j_vec)
                matrix[a][b] = dist
                matrix[b][a] = dist
    return matrix


def connlevel_sequence(metadata: dict[ConnectionKey, list[PackageInfo]], mapping: dict[ConnectionKey, int],
                       allLabels: list[str]):
    global totalConns
    inv_mapping = {v: k for k, v in mapping.items()}
    data = metadata
    timing = {}
    values = list(data.values())
    totalConns = len(values)
    keys = list(data.keys())
    distm = []
    labels = []
    ipmapping = []
    '''for i,v in data.items():
        fig = plt.figure(figsize=(10.0,9.0))
        ax = fig.add_subplot(111)
        ax.set_title(i)
        plt.plot([x[1] for x in v][:75], 'b')
        plt.plot([x[1] for x in v][:75], 'b.')
        cid = keys.index(i)
        plt.savefig('unzipped/malevol/data/connections/'+str(cid)+'.png')'''

    # save intermediate results

    addition = '-' + expname + '-' + str(thresh)

    # ----- start porting -------

    utils, r = None, None

    """
    -------------------
    Bytes
    -------------------
    """
    startb = time.time()

    filename = 'bytesDist' + addition + '.txt'
    if os.path.exists(filename):
        distm = []
        linecount = 0
        for line in open(filename, 'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                distm[linecount].append(float(e))
            linecount += 1

        for line in open('labels' + addition + '.txt', 'r').readlines():
            labels = [int(e) for e in line.split(' ')]

        print("found bytes.txt")
        distm = np.array(distm)
    else:
        print("Calculate packet size similarities:", flush=True)

        for a in range(len(data.values())):
            labels.append(mapping[keys[a]])
            ipmapping.append((mapping[keys[a]], inv_mapping[mapping[keys[a]]]))

        distm = distanceMatrixBytes(data, values)

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):  # len(data.values())): #range(10):
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")
        with open('labels' + addition + '.txt', 'w') as outfile:
            outfile.write(' '.join([str(l) for l in labels]) + '\n')
        with open('mapping' + addition + '.txt', 'w') as outfile:
            outfile.write(' '.join([str(l) for l in ipmapping]) + '\n')
    endb = time.time()
    print('Time of bytes: ', (endb - startb))
    ndistmB = []
    mini = distm.min()
    maxi = distm.max()

    for a in range(len(distm)):
        ndistmB.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / (maxi - mini)
            ndistmB[a].append(normed)

    """
    -------------------
    GAPS
    -------------------
    """
    startg = time.time()
    distm = []

    filename = 'gapsDist' + addition + '.txt'
    # Gap distances
    if os.path.exists(filename):
        linecount = 0
        for line in open(filename, 'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                try:
                    distm[linecount].append(float(e))
                except:
                    print("error on: " + e)
            linecount += 1

        # print distm
        print("found gaps.txt")
        distm = np.array(distm)
    else:
        print("Calculate gaps similarities", flush=True)
        distm = distanceMatrixGaps(data, values)

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):  # len(data.values())): #range(10):
                # print distm[a]
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    endg = time.time()
    print('gaps ', (endg - startg))
    ndistmG = []
    mini = distm.min()
    maxi = distm.max()

    for a in range(len(distm)):  # len(data.values())): #range(10):
        ndistmG.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / (maxi - mini)
            ndistmG[a].append(normed)

    """
    -------------------
    Source port
    -------------------
    """
    ndistmS = []
    distm = []

    starts = time.time()

    filename = 'sportDist' + addition + '.txt'
    same, diff = set(), set()
    if os.path.exists(filename):

        linecount = 0
        for line in open(filename, 'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                try:
                    distm[linecount].append(float(e))
                except:
                    print("error on: " + e)
            linecount += 1
        distm = np.array(distm)
        # print distm
        print("found sport.txt")
    else:
        print("Calculating source port similarities")
        distm = distanceMatrixSource(data, values)

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):  # len(data.values())): #range(10):
                # print distm[a]
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    ends = time.time()
    print('sport ', (ends - starts))

    mini = distm.min()
    maxi = distm.max()
    # print mini
    # print maxi
    # print "effective connections " + str(len(distm[0]))
    # print "effective connections  " + str(len(distm))

    for a in range(len(distm)):  # len(data.values())): #range(10):
        ndistmS.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / (maxi - mini)
            ndistmS[a].append(normed)

    """
    -------------------
    Destination port
    -------------------
    """
    ndistmD = []
    distm = []
    startd = time.time()

    filename = 'dportDist' + addition + '.txt'
    if os.path.exists(filename):

        linecount = 0
        for line in open(filename, 'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                try:
                    distm[linecount].append(float(e))
                except:
                    print("error on: " + e)
            linecount += 1

        # print distm
        print("found dport.txt")
        distm = np.array(distm)
    else:
        print("Calculating destination port similarities")
        distm = distanceMatrixDest(data, values)

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):  # len(data.values())): #range(10):
                # print distm[a]
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    endd = time.time()
    print('time dport ', (endd - startd))
    mini = distm.min()
    maxi = distm.max()
    # print mini
    # print maxi
    for a in range(len(distm)):  # len(data.values())): #range(10):
        ndistmD.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / (maxi - mini)
            ndistmD[a].append(normed)

    """
    Calculate average distances
    """
    ndistm = []
    print(np.array(ndistmB).shape)
    print(np.array(ndistmG).shape)
    print(np.array(ndistmD).shape)
    print(np.array(ndistmS).shape)
    for a in range(len(ndistmS)):  # len(data.values())): #range(10):

        ndistm.append([])
        for b in range(len(ndistmS)):
            ndistm[a].append((ndistmB[a][b] + ndistmG[a][b] + ndistmD[a][b] + ndistmS[a][b]) / 4.0)

    ndistm = np.array(ndistm)
    mini = ndistm.min()
    maxi = ndistm.max()
    finalMatrix = []
    for a in range(len(ndistm)):  # len(data.values())): #range(10):

        finalMatrix.append([])
        for b in range(len(ndistm)):
            normed = (ndistm[a][b] - mini) / (maxi - mini)
            finalMatrix[a].append(normed)

    finalMatrix = np.array(finalMatrix)

    print("Done with distance measurement")
    print("------------------------------\n")
    print("Start clustering: ")

    plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}

    # PROJECTION AND DIMENSIONALITY STUFF


    # def gridSearchAgglomerativeClustering(minK=2, maxK=100):
    #     smallestError = 100
    #     bestClusters = None
    #     bestProj = None
    #     data = []
    #     linkage = ["average", "complete", "ward"]
    #     for clusters in tqdm(range(minK, maxK)):
    #         for link in linkage:
    #             projection = dimensionalityRed(reductionMethod, 100, finalMatrix)
    #             model = AgglomerativeClustering(n_clusters=clusters, linkage=link)
    #             clu = model.fit(projection)
    #             # Calculate metrics and find optimization of parameters
    #             silhouette_score, purityErr, malwareErr, noiseErr, clusterErr = finalClusterSummary(finalMatrix,
    #                                                                                                 clu, labels,
    #                                                                                                 values,
    #                                                                                                 inv_mapping,
    #                                                                                                 False)
    #             error = purityErr + malwareErr + noiseErr + clusterErr + ((1 - silhouette_score) / 2)
    #             if error < smallestError:
    #                 smallestError = error
    #                 bestClusters = clusters
    #                 bestLinkage = link
    #                 bestProj = projection
    #                 data = [silhouette_score, purityErr, malwareErr, noiseErr, clusterErr, error]
    #     print("Best clusters:" + str(bestClusters))
    #     print("Best linkage:" + bestLinkage)
    #     return bestProj, bestClusters, bestLinkage, data

    # def gridSearchOptics(minCluster=20, maxCluster=21):
    #     smallestError = 100
    #     bestCluster = None
    #     bestProj = None
    #     data = []
    #
    #     for cluster in tqdm(range(minCluster, maxCluster)):
    #         projection = dimensionalityRed(reductionMethod, 100, finalMatrix)
    #         model = OPTICS(min_samples = cluster)
    #         clu = model.fit(projection)
    #         # Calculate metrics and find optimization of parameters
    #         silhouette_score, purityErr, malwareErr, noiseErr, clusterErr = finalClusterSummary(finalMatrix,
    #                                                                                             clu, labels,
    #                                                                                             values,
    #                                                                                             inv_mapping,
    #                                                                                             False)
    #         error = purityErr + malwareErr + noiseErr + clusterErr + ((1 - silhouette_score) / 2)
    #         if error < smallestError:
    #             smallestError = error
    #             bestCluster = cluster
    #             bestProj = projection
    #             data = [silhouette_score, purityErr, malwareErr, noiseErr, clusterErr, error]
    #     print("Best cluster: " + str(cluster))
    #     return bestProj, bestCluster, data
    #
    # def gridSearchKMedoids(minK=2, maxK=100):
    #     smallestError = 100
    #     bestClusters = None
    #     bestProj = None
    #     data = []
    #
    #     for clusters in tqdm(range(minK, maxK)):
    #         projection = dimensionalityRed(reductionMethod, 100, finalMatrix)
    #         model = KMedoids(n_clusters=clusters)
    #         clu = model.fit(projection)
    #         # Calculate metrics and find optimization of parameters
    #         silhouette_score, purityErr, malwareErr, noiseErr, clusterErr = finalClusterSummary(finalMatrix,
    #                                                                                             clu, labels,
    #                                                                                             values,
    #                                                                                             inv_mapping,
    #                                                                                             False)
    #         error = purityErr + malwareErr + noiseErr + clusterErr + ((1 - silhouette_score) / 2)
    #         if error < smallestError:
    #             smallestError = error
    #             bestClusters = clusters
    #             bestProj = projection
    #             data = [silhouette_score, purityErr, malwareErr, noiseErr, clusterErr, error]
    #     print("Best clusters:" + str(bestClusters))
    #     return bestProj, bestClusters, data
    #
    # """
    # Clustering expects a distance matrix, returns a ID list with all cluster labeling of the connections
    # """
    #
    # def gridSearch(minD=100, maxD=101, minSize=2, maxSize=25, minSample=2, maxSample=25):
    #     smallestError = 100
    #     bestSize = None
    #     bestSample = None
    #     bestProj = None
    #     data = []
    #     for dimensions in tqdm(range(minD, maxD)):
    #         projection = dimensionalityRed(reductionMethod, dimensions, finalMatrix)
    #         for size in tqdm(range(minSize, maxSize)):
    #             for sample in range(minSample, maxSample):
    #                 # Fit the model
    #                 model = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=sample)
    #                 clu = model.fit(projection)
    #
    #                 # Calculate metrics and find optimization of parameters
    #                 silhouette_score, purityErr, malwareErr, noiseErr, clusterErr = finalClusterSummary(finalMatrix,
    #                                                                                                     clu, labels,
    #                                                                                                     values,
    #                                                                                                     inv_mapping,
    #                                                                                                     False)
    #                 error = purityErr + malwareErr + noiseErr + clusterErr + ((1 - silhouette_score) / 2)
    #                 if error < smallestError:
    #                     smallestError = error
    #                     bestSize = size
    #                     bestSample = sample
    #                     bestProj = projection
    #                     data = [silhouette_score, purityErr, malwareErr, noiseErr, clusterErr, error]
    #
    #     print("Best projection dimension: " + str(bestProj.shape[1]))
    #     print("Best size: " + str(bestSize))
    #     print("Best sample: " + str(bestSample))
    #     return bestProj, bestSize, bestSample, data

    # Summary of grid search results
    #projection, size, sample, data0 = gridSearch()
    #projection, clusters, linkage, data0 = gridSearchAgglomerativeClustering()
    #projection, clusters, data0 = gridSearchKMedoids()
    #projection, cluster, data0 = gridSearchOptics()

    # Cluster on distance matrix / feature vectors of connetions where every feature is a distance to another connection

    reductionMethod = "none"
    projection = dimensionalityRed(reductionMethod, 100, finalMatrix)

    # model = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=10)
    # clu = model.fit(projection)  # final for citadel and dridex
    # model = AgglomerativeClustering(n_clusters=13, linkage="average")
    # clu = model.fit(projection)
    model = KMedoids(n_clusters=54)
    clu = model.fit(projection)
    # model = OPTICS(min_samples= 34)
    # clu = model.fit(projection)

    # Visualize the matrix and save result
    projection = visualizeProjection(reductionMethod, projection, allLabels, mapping, data, addition)

    joblib.dump(clu, 'model' + addition + '.pkl')
    # print "size: " + str(size) + "sample: " + str(sample)+ " silhouette: " +  str(silhouette_score(ndistm, clu.labels_, metric='precomputed'))

    print("num clusters: " + str(len(set(clu.labels_)) - 1))

    avg = 0.0
    for l in list(set(clu.labels_)):
        if l != -1:
            avg += sum([(1 if x == l else 0) for x in clu.labels_])
    print("avergae size of cluster:" + str(float(avg) / float(len(set(clu.labels_)) - 1)))
    print("samples in noise: " + str(sum([(1 if x == -1 else 0) for x in clu.labels_])))
    # clu.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    # plt.show()
    # clu.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    # plt.show()

    cols = ['royalblue', 'red', 'darksalmon', 'sienna', 'mediumpurple', 'palevioletred', 'plum', 'darkgreen',
            'lightseagreen', 'mediumvioletred', 'gold', 'navy', 'sandybrown', 'darkorchid', 'olivedrab', 'rosybrown',
            'maroon', 'deepskyblue', 'silver']
    pal = sns.color_palette(cols)  #

    extra_cols = len(set(clu.labels_)) - 18

    pal_extra = sns.color_palette('Paired', extra_cols)
    pal.extend(pal_extra)
    col = [pal[x] for x in clu.labels_]
    assert len(clu.labels_) == len(ndistm)

    #mem_col = [sns.desaturate(x, p) for x, p in zip(col, clu.probabilities_)]

    # plt.scatter(*projection.T, s=50, linewidth=0, c=col, alpha=0.2)

    # classes = ['Alexa', 'Hue', 'Somfy', 'malware']
    # print([(x, col[i]) for i,x in enumerate(classes)])
    for i, txt in enumerate(clu.labels_):  # mapping.keys()): #zip([x[:1] for x in mapping.keys()],clu.labels_)):

        realind = labels[i]
        name = inv_mapping[realind]
        '''thiscol = None
        thislab = None
        for cdx, cc in enumerate(classes):
            if cc in name:
                thiscol = col[cdx]
                thislab = cc
                break'''
        plt.scatter(projection.T[0][i], projection.T[1][i], color=col[i], alpha=0.6)
        if txt == -1:
            continue

        plt.annotate(txt, (projection.T[0][i], projection.T[1][i]), color=col[i], alpha=0.6)
        # plt.scatter(projection.T[0][i],projection.T[1][i], color=col[i], alpha=0.6)
        # plt.annotate(thislab, (projection.T[0][i],projection.T[1][i]), color=thiscol, alpha=0.2)

    plt.savefig("clustering-result" + addition)
    # plt.show()

    # writing csv file
    print("writing csv file")
    final_clusters = {}
    final_probs = {}
    for lab in set(clu.labels_):
        occ = [i for i, x in enumerate(clu.labels_) if x == lab]
        #final_probs[lab] = [x for i, x in zip(clu.labels_, clu.probabilities_) if i == lab]
        print("cluster: " + str(lab) + " num items: " + str(len([labels[x] for x in occ])))
        final_clusters[lab] = [labels[x] for x in occ]

    csv_file = 'clusters' + addition + '.csv'
    outfile = open(csv_file, 'w')
    outfile.write("clusnum,connnum,probability,class,filename,srcip,dstip\n")

    for n, clus in final_clusters.items():
        # print "cluster numbeR: " + str(n)
        #
        for idx, el in enumerate([inv_mapping[x] for x in clus]):
            filename = el.pcap
            # outfile.write(
            #     str(n) + "," + str(mapping[el]) + "," + str(final_probs[n][idx]) + "," + str(filename) +
            #     "," + str(el.sourceIp) + "," + str(el.destIp) + ":" + str(el.window) +"\n")
            outfile.write(
                str(n) + "," + str(mapping[el]) + "," + "," + str(filename) +
                "," + str(el.sourceIp) + "," + str(el.destIp) + ":" + str(el.window) +"\n")

    outfile.close()

    silhouette_score, purityErr, malwareErr, noiseErr, clusterErr = finalClusterSummary(finalMatrix, clu, labels,
                                                                                        values, inv_mapping, True)
    error = purityErr + malwareErr + noiseErr + clusterErr + (1 - silhouette_score) / 2
    print("Total error:" + str(error))
    data1 = [silhouette_score, purityErr, malwareErr, noiseErr, clusterErr, error]

    # Plot the grid search results on HDBSCAN
    # dataa = [data0, data1]
    # labelss = ['silhouetteScore', 'purityErr', 'malwareErr', 'noiseErr', 'completenessErr', 'totalErr']
    # X = np.arange(6)
    # fig, ax = plt.subplots()
    # ax.bar(X - 0.375 / 2, dataa[0], color='#ffde59', width=0.375, label='HDBScan(19,1)')
    # ax.bar(X + 0.375 / 2, dataa[1], color='#ff914d', width=0.375, label='HDBScan(7,7)')
    # ax.set_ylabel('Error value')
    # ax.set_title('Result of grid search based parameter optimization of HDBScan')
    # ax.set_xticks(X)
    # ax.set_xticklabels(labelss)
    # ax.legend()
    # plt.savefig("RESULTS.png")

    print("-------------------------------------")
    # Making tree
    print('Producing DAG with relationships between pcaps')
    clusters = {}
    numclus = len(set(clu.labels_))
    with open(csv_file, 'r') as f1:
        reader = csv.reader(f1, delimiter=',')
        for i, line in enumerate(reader):  # f1.readlines()[1:]:
            if i > 0:
                if line[4] not in clusters.keys():
                    clusters[line[4]] = []
                clusters[line[4]].append((line[3], line[0]))  # classname, cluster#
    # print(clusters)
    f1.close()

    array = [str(x) for x in range(numclus)]
    array.append("-1")

    treeprep = dict()
    for filename, val in clusters.items():
        arr = [0] * numclus
        for fam, clus in val:
            ind = array.index(clus)
            arr[ind] = 1
        # print(filename, )
        mas = ''.join([str(x) for x in arr[:-1]])
        famname = fam
        print(filename + "\t" + fam + "\t" + ''.join([str(x) for x in arr[:-1]]))
        if mas not in treeprep.keys():
            treeprep[mas] = dict()
        if famname not in treeprep[mas].keys():
            treeprep[mas][famname] = set()
        treeprep[mas][famname].add(str(filename))

    f2 = open('mas-details' + addition + '.csv', 'w')
    for k, v in treeprep.items():
        for kv, vv in v.items():
            # print(k, str(kv), (vv))
            f2.write(str(k) + ';' + str(kv) + ';' + str(len(vv)) + '\n')
    f2.close()

    with open('mas-details' + addition + '.csv', 'rU') as f3:
        csv_reader = csv.reader(f3, delimiter=';')

        graph = {}

        names = {}
        for line in csv_reader:
            graph[line[0]] = set()
            if line[0] not in names.keys():
                names[line[0]] = []
            names[line[0]].append(line[1] + "(" + line[2] + ")")

        zeros = ''.join(['0'] * (numclus - 1))
        if zeros not in graph.keys():
            graph[zeros] = set()

        ulist = graph.keys()
        # print(len(ulist), ulist)
        covered = set()
        next = deque()

        specials = []

        next.append(zeros)

        while (len(next) > 0):
            # print(graph)
            l1 = next.popleft()
            covered.add(l1)
            for l2 in ulist:
                # print(l1, l2, difference(l1,l2))
                if l2 not in covered and difference(l1, l2) == 1:
                    graph[l1].add(l2)

                    if l2 not in next:
                        next.append(l2)

        # keys = graph.keys()
        val = set()
        for v in graph.values():
            val.update(v)

        notmain = [x for x in ulist if x not in val]
        notmain.remove(zeros)
        nums = [sum([int(y) for y in x]) for x in notmain]
        notmain = [x for _, x in sorted(zip(nums, notmain))]

        specials = notmain
        # print(notmain)
        # print(len(notmain))

        extras = set()

        for nm in notmain:
            comp = set()
            comp.update(val)
            comp.update(extras)

            mindist = 1000
            minli1, minli2 = None, None
            for l in comp:
                if nm != l:
                    diff = difference(nm, l)
                    if diff < mindist:
                        mindist = diff
                        minli = l

            diffbase = difference(nm, zeros)
            # print('diffs', nm, 'extra', mindist, 'with root', diffbase)
            if diffbase <= mindist:
                mindist = diffbase
                minli = zeros
                # print('replaced')

            num1 = sum([int(s) for s in nm])
            num2 = sum([int(s) for s in minli])
            if num1 < num2:
                graph[nm].add(minli)
            else:
                graph[minli].add(nm)

            extras.add(nm)

        # keys = graph.keys()
        val = set()
        for v in graph.values():
            val.update(v)
            f2 = open('relation-tree' + addition + '.dot', 'w')
            f2.write("digraph dag {\n")
            f2.write("rankdir=LR;\n")
            num = 0
            for idx, li in names.items():
                text = ''
                # print(idx)
                name = str(idx) + '\n'

                for l in li:
                    name += l + ',\n'
                # print(str(idx) + " [label=\""+str(num)+"\"]")
                if idx not in specials:
                    # print(str(idx) + " [label=\""+name+"\"]")
                    text = str(idx) + " [label=\"" + name + "\" , shape=box;]"
                else:  # treat in a special way. For now, leaving intact
                    # print(str(idx) + " [style=\"filled\" fillcolor=\"red\" label=\""+name+"\"]")
                    text = str(idx) + " [shape=box label=\"" + name + "\"]"

                f2.write(text)
                f2.write('\n')
            for k, v in graph.items():
                for vi in v:
                    f2.write(str(k) + "->" + str(vi))
                    f2.write('\n')
                    # print(k+"->"+vi)
            f2.write("}")
            f2.close()
        # Rendering DAG
        print('Rendering DAG -- needs graphviz dot')
        try:
            os.system('dot -Tpng relation-tree' + addition + '.dot -o DAG' + addition + '.png')
            print('Done')
        except:
            print('Failed')
            pass

    # temporal heatmaps start

    print("writing temporal heatmaps")
    # print("prob: ", clu.probabilities_)
    if not os.path.exists('figs' + addition + '/'):
        os.mkdir('figs' + addition + '/')
        os.mkdir('figs' + addition + '/bytes')
        os.mkdir('figs' + addition + '/gaps')
        os.mkdir('figs' + addition + '/sport')
        os.mkdir('figs' + addition + '/dport')

    actlabels = []
    for a in range(len(values)):  # range(10):
        actlabels.append(mapping[keys[a]])

    clusterinfo = {}
    seqclufile = csv_file
    lines = []
    lines = open(seqclufile).readlines()[1:]

    for line in lines:
        li = line.split(",")  # clusnum, connnum, prob, srcip, dstip
        # if li[0] == '-1':
        #    continue

        pcap = li[3].replace(".pcap.pkl", "") + "="
        srcip = li[4]
        dstIpWindow = li[5]
        has = int(li[1])

        name = str('%12s->%12s=:%12s' % (pcap, srcip, dstIpWindow))
        if li[0] not in clusterinfo.keys():
            clusterinfo[li[0]] = []
        clusterinfo[li[0]].append((has, name))
    print("rendering ... ")

    sns.set(font_scale=0.9)
    matplotlib.rcParams.update({'font.size': 10})
    for names, sname, q in [("Packet sizes", "bytes", 1), ("Interval", "gaps", 0), ("Source Port", "sport", 2),
                            ("Dest. Port", "dport", 3)]:
        for clusnum, cluster in clusterinfo.items():
            items = [int(x[0]) for x in cluster]
            if len(items) == 1:
                continue
            labels = [x[1] for x in cluster]

            acha = [actlabels.index(int(x[0])) for x in cluster]

            blah = [values[a] for a in acha]

            dataf = []

            for b in blah:
                dataf.append([x.mapIndex(q) for x in b])

            df = pd.DataFrame(dataf, index=labels)

            g = sns.clustermap(df, xticklabels=False, col_cluster=False)  # , vmin= minb, vmax=maxb)
            ind = g.dendrogram_row.reordered_ind
            fig = plt.figure(figsize=(15.0, 9.0))
            plt.suptitle("Exp: " + expname + " | Cluster: " + clusnum + " | Feature: " + names)
            ax = fig.add_subplot(111)
            datanew = []
            labelsnew = []
            lol = []
            for it in ind:
                labelsnew.append(labels[it])
                # print labels[it]

                # print cluster[[x[1] for x in cluster].index(labels[it])][0]
                lol.append(cluster[[x[1] for x in cluster].index(labels[it])][0])
            # print len(labelsnew)
            # print len(lol)
            acha = [actlabels.index(int(x)) for x in lol]
            # print acha
            blah = [values[a] for a in acha]

            dataf = []

            for b in blah:
                dataf.append([x.mapIndex(q) for x in b])
            df = pd.DataFrame(dataf, index=labelsnew)
            g = sns.heatmap(df, xticklabels=False)
            plt.setp(g.get_yticklabels(), rotation=0)
            plt.subplots_adjust(top=0.92, bottom=0.02, left=0.25, right=1, hspace=0.94)
            plt.savefig("figs" + addition + "/" + sname + "/" + clusnum)


# Metrics
def clusterPurityError(summaries):
    totalError = 0
    nClusters = len(summaries)
    for summary in summaries:
        perc = summary['percentage']
        # Ignore if noise cluster
        if 'noise' in summary:
            continue
        if perc <= 0.5:
            error = perc
        else:
            error = 1 - perc
        totalError += error * 2
    return totalError / nClusters


def malwarePurityError(summaries):
    totalError = 0
    maliciousClusters = 0
    behaviours = {"PartOfAHorizontalPortScan", "Okiru", "Attack", "DDoS", "C&C", "FileDownload", "HeartBeat"}
    for summary in summaries:
        # Ignore noise cluster
        if 'noise' in summary:
            continue
        labels = summary['labels']
        if len(labels) > 0:
            maliciousClusters += 1
        # If this cluster is malicious
        if len(labels) > 1:
            downloads = 0
            for label in labels:
                if "FileDownload" in label:
                    downloads += 1
            downloads = max(0, downloads - 1)
            error = (len(labels) - downloads) / len(behaviours)
            totalError += error
    return totalError / maliciousClusters


def completenessError(summaries):
    clusterBehaviours = defaultdict(int)
    nClusters = len(summaries)
    totalError = 0
    for summary in summaries:
        # Ignore noise cluster
        if 'noise' in summary:
            continue
        if summary['benign'] > 0:
            clusterBehaviours['benign'] += 1
        labels = summary['labels']
        for label in labels:
            clusterBehaviours[label] += 1
    for value in clusterBehaviours.values():
        # If there is a label that is in more than 1 cluster:
        if value > 1:
            error = (value / len(clusterBehaviours)) / 10
            totalError += error
    return totalError


def noiseError(summaries):
    for summary in summaries:
        if "noise" in summary:
            return (summary['total'] / 20) / totalConns
    return 0


def allMetrics(finalMatrix, clusterResults, summaries, toPrint):
    silhouetteScore = silhouette_score(finalMatrix, clusterResults, metric='precomputed')
    purityErr = clusterPurityError(summaries)
    malwareErr = malwarePurityError(summaries)
    noiseErr = noiseError(summaries)
    completenessErr = completenessError(summaries)

    if toPrint:
        print("Avg silhouette score is:" + str(silhouetteScore))
        print("Avg silhouette error is : " + str((1 - silhouetteScore) / 2))
        print("Cluster purity error is: " + str(purityErr))
        print("Cluster malware error is: " + str(malwareErr))
        print("Noise error is: " + str(noiseErr))
        print("Completeness error is: " + str(completenessErr))
    return silhouetteScore, purityErr, malwareErr, noiseErr, completenessErr


# Distance matrix, clustering results
def finalClusterSummary(finalMatrix, clu, labels, values, inv_mapping, toPrint):
    finalClusters = {}
    for lab in set(clu.labels_):
        occ = [i for i, x in enumerate(clu.labels_) if x == lab]
        # print("cluster: " + str(lab) + " num items: " + str(len([labels[x] for x in occ])))
        finalClusters[lab] = [labels[x] for x in occ]

    summaries = []
    # n ==  cluster number, cluster = items in cluster
    for cluster, connections in finalClusters.items():
        summary = {'labels': set(), 'total': thresh * len(connections), 'malicious': 0, 'benign': 0, 'percentage': 0}
        for connectionNumber in connections:
            updateSummary(summary, values[connectionNumber], inv_mapping[connectionNumber])
        percentage = summary['percentage']
        # If this is a noise cluster:
        if cluster == -1:
            summary.update({"noise": "True"})
        summaries.append(summary)
        if percentage > 0 and toPrint:
            print(
                f"cluster {cluster} is {round(percentage * 100, 2)}% malicious, contains following labels: {','.join(summary['labels'])}, connections: {len(connections)}")
        elif toPrint:
            print(f"cluster {cluster} does not contain any malicious packages, connections: {len(connections)}")

    if toPrint:
        percentages = [x['percentage'] for x in summaries]

    return allMetrics(finalMatrix, clu.labels_, summaries, toPrint)


def updateSummary(summary: dict, packages: list[PackageInfo], connection: str):
    percentage = summary['percentage']
    malicious = summary['malicious']
    for package in packages:
        if package.connectionLabel != '-':
            summary['malicious'] += 1
            summary['labels'].add(package.connectionLabel)
        else:
            summary['benign'] += 1

    summary.update({'percentage': summary['malicious'] / summary['total']})
    # malpercent = round(((summary['malicious'] - malicious) / 20) * 100)
    # if summary['percentage'] > percentage:
    #     print("Connection " + connection + " is " + str(malpercent) + "% malicious!")


# Returns a projection of the lower dimension space
def dimensionalityRed(method, dimensions, matrix):
    RS = 3072018
    if "pca" in method:
        return PCA(n_components=dimensions).fit_transform(matrix)
    elif "tsne" in method:
        return TSNE(method='exact', random_state=RS).fit_transform(matrix)
    elif "umap" in method:
        return umap.UMAP().fit_transform(matrix)
    elif "mds" in method:
        return MDS(metric=True, n_components=dimensions).fit_transform(matrix)
    elif "none" in method:
        return matrix


# Applies a TSNE 2d projection of the data, and plots it with the corresponding malicious labels.
def visualizeProjection(method, projection, allLabels, mapping, data, addition):
    RS = 2021
    projection = TSNE(random_state=RS).fit_transform(projection)
    colors = sns.color_palette("hls", len(allLabels))
    colorDict = dict(zip(allLabels, colors))
    plt.figure(figsize=(10, 10))
    for label in allLabels:
        # Gives all connection numbers corresponding to a specific label
        connNumbers = [mapping[conn] for conn in data if data[conn][0].connectionLabel == label]
        subset = projection[connNumbers]
        x = [row[0] for row in subset]
        y = [row[1] for row in subset]
        if label == "-":
            plt.scatter(x, y, label="Benign", color=colorDict[label])
        else:
            plt.scatter(x, y, label=label, color=colorDict[label])
    plt.legend()
    plt.savefig("Visualization result of " + method + addition)
    plt.clf()
    return projection


def inet_to_str(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def readpcap(filename, labels):
    print("Reading", os.path.basename(filename))
    counter = 0

    connections = {}
    previousTimestamp = {}

    f = open(filename, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, pkt in pcap:
        counter += 1

        try:
            eth = dpkt.ethernet.Ethernet(pkt)
        except:
            continue

        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
            continue

        ip = eth.data

        src_ip = inet_to_str(ip.src)
        dst_ip = inet_to_str(ip.dst)

        key = (src_ip, dst_ip)

        timestamp = datetime.datetime.utcfromtimestamp(ts)

        if key in previousTimestamp:
            gap = (timestamp - previousTimestamp[key]).microseconds / 1000
        else:
            gap = 0

        previousTimestamp[key] = timestamp

        sport = 0
        dport = 0

        try:
            if ip.p == dpkt.ip.IP_PROTO_TCP or ip.p == dpkt.ip.IP_PROTO_UDP:
                sport = ip.data.sport
                dport = ip.data.dport
        except:
            continue

        if key not in connections.keys():
            connections[key] = []

        label = labels.get(hash((src_ip, dst_ip, sport, dport))) or labels.get(hash((dst_ip, src_ip, dport, sport)))
        connections[key].append((gap, ip.len, ip.p, sport, dport, label))

    print(os.path.basename(filename), " num connections: ", len(connections))

    todel = []
    print('Before cleanup: Total packets: ', len(connections), ' connections.')
    for i, v in connections.items():  # clean it up
        if len(v) < thresh:
            todel.append(i)

    for item in todel:
        del connections[item]

    print("Remaining connections after clean up ", len(connections))

    return connections


"""
Returns all connections with key = pcap, sourceIp, destIp
"""


# def readLabeledData() -> dict[tuple[str, str, str], list[PackageInfo]] :
#     connections = {}
#     files = glob.glob(sys.argv[2] + "/*.pkl")
#     print("Reading all pickle files", flush=True)
#     for f in files:
#         pcapName = (os.path.basename(f),)
#         newConns = readPcapPkl(f)
#         connsWithPcap = {pcapName + k:v for (k,v) in newConns.items()}
#         print(len(connsWithPcap))
#         connections.update(connsWithPcap)
#
#     print(len(connections))
#     return connections


def readPcapPkl(file) -> dict[tuple[str, str], list[PackageInfo]]:
    with open(file, 'rb') as file:
        connections = pickle.load(file)

    return connections


def percentMalware(connsLabeled, key):
    length = len(connsLabeled[key])
    counter = 0
    for packet in connsLabeled[key]:
        if packet.isMalicious:
            print("ALERT! Connection " + key + "(Source: " + str(packet.sourcePort) + ", " + str(
                packet.destPort) + ") contains " + packet.label + " as labeled malicious behaviour")
            counter += 1
    return (counter / length) * 100


"""
This method filters all connections to contain a balanced dataset (malware vs benign ratio)
It also uses a sliding window to split a connection into multiple.
"""


def readLabeledData(maxThresh=100):
    # Reads the labeled pickle files
    mappingIndex = 0
    meta = {}
    mapping = {}
    totalLabels = defaultdict(int)
    files = glob.glob(sys.argv[2] + "/*.pkl")
    for f in files:
        pcapName = (os.path.basename(f),)
        newConns = readPcapPkl(f)
        connections = {pcapName + k: v for (k, v) in newConns.items()}
        connectionItems: list[tuple[str, str, str], list[PackageInfo]] = list(connections.items())
        random.shuffle(connectionItems)

        #connectionItems = connectionItems[0: (len(connectionItems) // 2)]
        connectionItems = connectionItems[(len(connectionItems) // 2):]
        print(len(connectionItems))
        selectedLabelsPerFile = defaultdict(int)

        for k, v in connectionItems:
            wantedWindow = getWantedWindow(v)

            for window in wantedWindow:
                key = ConnectionKey(k[0], k[1], k[2], window)
                selection = v[thresh * window: thresh * (window + 1)]
                labels = set()
                for package in selection:
                    labels.add(package.connectionLabel)

                if len(labels) != 1:
                    continue

                label = labels.pop()

                if selectedLabelsPerFile[label] >= maxThresh:
                    continue
                selectedLabelsPerFile[label] += 1
                mapping[key] = mappingIndex
                mappingIndex += 1
                meta[key] = selection

        print(str(selectedLabelsPerFile.items()))
        for k, v in selectedLabelsPerFile.items():
            totalLabels[k] += v

    labels = list(totalLabels.keys())
    labels.sort()
    print(totalLabels)
    print('Done reading labeled data..')
    print('Collective surviving connections ', len(mapping))
    print("----------------------------------------\n")

    connlevel_sequence(meta, mapping, labels)


def getWantedWindow(v):
    amountOfPackages = len(v)
    windowRange = list(range(amountOfPackages // thresh))
    possibleWindows = len(windowRange)

    if possibleWindows == 1:
        return [0]
    elif possibleWindows == 2:
        return [0, 1]
    else:
        wantedWindow = windowRange[:1] + windowRange[-1:]
        wantedWindow += random.sample(windowRange[1:-1], min(len(windowRange) - 2, 8))
        return wantedWindow


def readfile():
    startf = time.time()
    mapping = {}
    print('About to read pcap...')
    data, connections = readpcap(sys.argv[2])
    print('Done reading pcaps...')
    if len(connections.items()) < 1:
        return

    endf = time.time()
    print('file reading ', (endf - startf))
    fno = 0
    meta = {}
    nconnections = {}
    print("Average conn length: ", np.mean([len(x) for i, x in connections.items()]))
    print("Minimum conn length: ", np.min([len(x) for i, x in connections.items()]))
    print("Maximum conn length: ", np.max([len(x) for i, x in connections.items()]))
    # print("num connections survived ", len(connections))
    # print(sum([1 for i,x in connections.items() if len(x)>=50]))
    for i, v in connections.items():
        name = i[0] + "->" + i[1]
        mapping[name] = fno
        fno += 1
        meta[name] = v

        '''fig = plt.figure()
        plt.title(''+name)
        plt.plot([x[0] for x in v], 'r')
        plt.plot([x[0] for x in v], 'r.')
        plt.savefig('figs/'+str(mapping[name])+'.png')'''
    print('Surviving connections ', len(meta))
    startc = time.time()
    connlevel_sequence(meta, mapping)
    endc = time.time()
    print('Total time ', (endc - startc))


def main():
    if sys.argv[1] == 'file':
        readfile()

    elif sys.argv[1] == 'folder':
        readLabeledData()
    else:
        print('incomplete command')


main()
