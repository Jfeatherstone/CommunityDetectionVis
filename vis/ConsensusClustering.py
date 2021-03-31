import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi

import os

from .Visualization import isInsideHull, intersection, genRandomColors, drawPoints, drawLines, mapPointToColor
from .CommunityWise import patchCommunityIndices, concaveHull, vorToCommunityBounds, assignCommunities, calculateCommunityCenters

import networkx as nx
import community as community_louvain

def calculateConsensusMatrix(communityAssignments):
    # Mostly just for testing, so I can use a single community detection
    if len(np.shape(communityAssignments)) == 1:
        communityAssignments = [communityAssignments]

    numPoints = len(communityAssignments[0])
    consensusMatrix = np.zeros([numPoints, numPoints])

    # Iterate over every detection
    # Quite inefficient, but very simple at least
    for i in range(len(communityAssignments)):
        for j in range(numPoints):
            for k in range(numPoints):
                consensusMatrix[j,k] += int(communityAssignments[i][j] == communityAssignments[i][k])


    return consensusMatrix/len(communityAssignments)

def consensusCluster(pointArr, communityArr, numCommunityDetections=5, numHiddenDetections=3, gridPoints=60, resolutionCenter=.75, resolutionVar=.05, figOutputFolder=None):

    # If we want to save the figures, make sure the directory exists
    if figOutputFolder != None:
        if not os.path.isdir(figOutputFolder):
            os.mkdir(figOutputFolder)

    # Create a uniform grid of points
    # Not exactly the most efficient method to find the extreme points across
    # all configurations, but it works well enough
    flattenedX = []
    flattenedY = []

    for i in range(len(communityArr)):
        for j in range(len(pointArr[i])):
            flattenedX.append(pointArr[i][j,0])
            flattenedY.append(pointArr[i][j,1])

    uniformX = np.linspace(min(flattenedX), max(flattenedX), gridPoints)
    uniformY = np.linspace(min(flattenedY), max(flattenedY), gridPoints)

    uniformGridPoints = np.zeros([gridPoints, gridPoints, 2])

    for i in range(gridPoints):
        for j in range(gridPoints):
            uniformGridPoints[i,j] = [uniformX[i], uniformY[j]]
            
    uniformGridPoints = uniformGridPoints.reshape([gridPoints*gridPoints, 2])

    hullLines = []
    communityBoundaries = []

    uniformPointArr = []
    uniformCommunityArr = []

    for i in range(len(communityArr)):
        # Calculate the boundaries for each detection
        # NOTE: This may take a while, about 1-2 minutes per detection for 50 gridPoints
        currVor = Voronoi(pointArr[i])
        currHullLines, currHullLineCommunities = concaveHull(pointArr[i], communityArr[i])
        currCommunityBoundaries = vorToCommunityBounds(currVor, communityArr[i], currHullLines, currHullLineCommunities)
        
        # Make sure that there aren't any weird things going on with this detection
        # Sometimes you will get artifacts of something (idk what) so we just throw those away\
        badDetection = False
        for j in range(len(currCommunityBoundaries)):
            if len(currCommunityBoundaries[j]) == 0:
                badDetection = True
                break
        if badDetection:
            print(f'Warning: bad detection at index {i}')
            continue
            
        currUniformPoints, currUniformCommunities = assignCommunities(uniformGridPoints, currCommunityBoundaries, currHullLines)
        
        # And save some things
        hullLines.append(currHullLines)
        communityBoundaries.append(currCommunityBoundaries)
        uniformPointArr.append(currUniformPoints)
        uniformCommunityArr.append(currUniformCommunities)

    # We also have to make sure that each uniform set of points
    # is exactly the same, since the slight variations in boundary sizes
    # could make a difference
    pointCounts = [len(uniformPointArr[i]) for i in range(len(uniformPointArr))]
    if len(np.unique(pointCounts)) > 1:
        print(pointCounts)
        # Unfortunately, numpy is really annoying when it comes to deleting
        # elements of multidimensional arrays, and you can't easily delete
        # an entire sub array
        # Because of this, we convert our point arrays to regular lists,
        # use the regular python del method, and then convert back after
        for i in range(len(uniformPointArr)):
            uniformPointArr[i] = list(uniformPointArr[i])
            uniformCommunityArr[i] = list(uniformCommunityArr[i])
            
            # The easiest way to determine which points are valid and which aren't invovles
            # checking each point to make sure it is in every hull
            # This is much faster than trying to compare the list of points themselves
            # O(n*m) vs O(n^m) (n = num points, m = num detections)
            for j in range(len(uniformPointArr[i])-1, -1, -1):
                # Check all of the points are in all of the hulls
                for k in range(len(hullLines)):
                    if not isInsideHull(uniformPointArr[i][j], hullLines[k]):
                        del uniformPointArr[i][j]
                        del uniformCommunityArr[i][j]
                        break

            # And convert back to numpy arrays
            uniformPointArr[i] = np.array(uniformPointArr[i])
            uniformCommunityArr[i] = np.array(uniformCommunityArr[i])
            # Just in case we removed an entire community near the edges
            # (unlikely but possible) we'll repatch the indices as well
            uniformCommunityArr[i] = patchCommunityIndices(uniformCommunityArr[i])


    pointCounts = [len(uniformPointArr[i]) for i in range(len(uniformPointArr))]
    communityCounts = [len(uniformCommunityArr[i]) for i in range(len(uniformCommunityArr))]
    print(pointCounts)
    print(communityCounts)

    if figOutputFolder != None:
        #randomColors = [genRandomColors(max(uniformCommunityArr[i])+1) for i in range(len(uniformCommunityArr))]
        colors = [[mapPointToColor(p) for p in calculateCommunityCenters(communityBoundaries[i])] for i in range(len(communityBoundaries))]
        fig, ax = plt.subplots(1, len(uniformCommunityArr), figsize=(len(uniformCommunityArr)*5, 3))

        for i in range(len(uniformCommunityArr)):
            drawPoints(ax[i], uniformPointArr[i], uniformCommunityArr[i], colors[i], s=5)
            for j in range(len(communityBoundaries[i])):
                drawLines(ax[i], communityBoundaries[i][j], opacity=.2)
        fig.tight_layout()
        plt.savefig(f'{figOutputFolder}/input_detections.png')
        plt.show()


    for j in range(numHiddenDetections):
        # Now calculate the consensus matrix and perform community detection
        consensusMatrix = calculateConsensusMatrix(uniformCommunityArr)

        # plt.pcolor(consensusMatrix)
        # plt.colorbar()
        # plt.show()

        # Now we can create a graph from the consensus matrix
        graph = nx.from_numpy_matrix(consensusMatrix)
        finalPoints = uniformPointArr[0]

        partitionArr = []
        genlouvainResolutions = np.random.normal(resolutionCenter, resolutionVar, size=[numCommunityDetections])
        for i in range(numCommunityDetections):
            # Perform the community detection
            currPartition = community_louvain.best_partition(graph, resolution=genlouvainResolutions[i], randomize=True)
            currPartition = list(currPartition.values())
            currPartition = patchCommunityIndices(currPartition)
            partitionArr.append(currPartition)

        if figOutputFolder != None:
            #randomColors = [genRandomColors(max(partitionArr[i])+1) for i in range(len(partitionArr))]
            colors = []
            for k in range(len(partitionArr)):
                centers = []
                for i in range(max(partitionArr[k])+1):
                    pointsInThisCommunity = finalPoints[partitionArr[k] == i]
                    centers.append([np.mean(pointsInThisCommunity[:,0]), np.mean(pointsInThisCommunity[:,1])])

                colors.append([mapPointToColor(c) for c in centers])

            fig, ax = plt.subplots(1, len(partitionArr), figsize=(len(partitionArr)*5, 3))

            for i in range(len(partitionArr)):
                drawPoints(ax[i], finalPoints, partitionArr[i], colors[i], s=5)
            fig.tight_layout()
            plt.savefig(f'{figOutputFolder}/hidden_detection_{j}.png')
            plt.show()

        uniformCommunityArr = partitionArr

    # Perform community detection one final time, and return the results 
    consensusMatrix = calculateConsensusMatrix(uniformCommunityArr)
    graph = nx.from_numpy_matrix(consensusMatrix)

    partition = community_louvain.best_partition(graph, resolution=genlouvainResolutions[i], randomize=True)
    partition = list(partition.values())
    partition = patchCommunityIndices(partition)

    if figOutputFolder != None:
    
        centers = []
        for i in range(max(partition)+1):
            pointsInThisCommunity = finalPoints[partition == i]
            centers.append([np.mean(pointsInThisCommunity[:,0]), np.mean(pointsInThisCommunity[:,1])])

        colors = [mapPointToColor(c) for c in centers]
        #randomColors = genRandomColors(max(partition)+1)
        fig, ax = plt.subplots(1, 1)

        drawPoints(ax, finalPoints, partition, colors, s=5)
        fig.tight_layout()
        plt.savefig(f'{figOutputFolder}/final_detection.png')
        plt.show()

    return finalPoints, partition
