import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi

import os

from .Visualization import isInsideHull, intersection, genRandomColors, drawPoints, drawLines, mapPointToColor
from .CommunityWise import patchCommunityIndices, concaveHull, vorToCommunityBounds, assignCommunities, calculateCommunityCenters, assignCommunitiesAndNeighbors

import networkx as nx
import community as community_louvain

from sklearn.neighbors import KDTree

def _dist(p1, p2):
    """
    Simple euclidean distance between two points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculateConsensusMatrix(communityAssignments):
    """
    Calculate the exact consensus matrix for a set of community detections (over the same set of points).

    Parameters
    ----------

    communityAssignments : list or numpy.ndarray
        A list of or array of dimensions (M, N), where:

        M : The number of community detections to create the matrix from

        N : The number of points assigned to communities

        The entry at [i,j] should be the community assignment for the jth point in the ith detection
       

    Returns
    -------
    numpy.ndarray : An (N, N) matrix of the fraction of detections that each particle is placed in the same community

    """
    # Mostly just for testing, so I can use a single community detection
    if len(np.shape(communityAssignments)) == 1:
        communityAssignments = [communityAssignments]

    numPoints = len(communityAssignments[0])
    consensusMatrix = np.zeros([numPoints, numPoints])

    # Iterate over every detection
    # Quite inefficient, but very simple at least
    for i in range(numPoints):
        for j in range(numPoints):
            consensusMatrix[i,j] = len([0 for k in range(len(communityAssignments)) if communityAssignments[k][i] == communityAssignments[k][j]])


    return consensusMatrix/len(communityAssignments)

def findNodeNeighbors(gridPositions, numNeighbors=8):
    """
    Find the nearest neighbors of each point in the provided grid. Calculation is done
    using a kd-tree, making the computation incredibly fast.

    Parameters
    ----------

    gridPositions : numpy.ndarray or list
        A list of N points; shape (N, 2).
       
    numNeighbors : int
        The number of neighbors to find for each point

    Returns
    -------

    list : A list of indices for each point of the other points that are its neighbors

    """
    # Use a KD Tree to find the neighbors (really) efficiently
    # I don't actually know the complexity of searching a kd tree
    # but judging by the time it takes to do it, it very well could be O(1)
    kdTree = KDTree(gridPositions, leaf_size=2)
    dist, ind = kdTree.query(gridPositions, k=numNeighbors+1)
    # The first index will always be the point itself, so we ignore that        
    return ind[:,1:]


def calculateFastConsensusMatrix(communityAssignments, neighborIndices, m=None):
    """
    Calculate the "fast" consensus matrix as described in:

    Tandon, A., Albeshri, A., Thayananthan, V., Alhalabi, W., & Fortunato, S. (2019). Fast consensus clustering in complex networks. Physical Review E, 99(4), 042301. [https://doi.org/10.1103/PhysRevE.99.042301](https://doi.org/10.1103/PhysRevE.99.042301)

    Procedure involves calculating only the nearest neighbor elements of the full consensus matrix,
    and then randomly sampling points to fill in neighbor-neighbor connections.

    Parameters
    ----------

    communityAssignments : list or numpy.ndarray
        A list of or array of dimensions (M, N), where:

        M : The number of community detections to create the matrix from

        N : The number of points assigned to communities

        The entry at [i,j] should be the community assignment for the jth point in the ith detection
       
    neighborIndices : list
        A list of the point indices corresponding to neighbors of each point. Any number of neighbors can be
        specified.

    m : int
        The number of neighbor-neighbor interactions to fill in using Monte-Carlo methods. Will become comparable
        to the exact calculation if this value is equal to the number of points times the number of neighbors (order
        of magnitude estimate).

    Returns
    -------
    numpy.ndarray : An (N, N) matrix of the fraction of detections that each particle is placed in the same community
    """
    numPoints = len(communityAssignments[0])
    consensusMatrix = np.zeros([numPoints, numPoints])

    if m == None:
        m = numPoints

    for i in range(numPoints):
        for nn in neighborIndices[i]:
            consensusMatrix[i][nn] = len([0 for k in range(len(communityAssignments)) if communityAssignments[k][i] == communityAssignments[k][nn]])
               
    for i in range(m):
        nodeSelection = np.random.randint(numPoints)
        n1, n2 = np.random.choice(neighborIndices[nodeSelection], size=2)
        
        if consensusMatrix[n1][n2] == 0:
            consensusMatrix[n1][n2] = len([0 for k in range(len(communityAssignments)) if communityAssignments[k][n1] == communityAssignments[k][n2]])

    consensusMatrix /= len(communityAssignments)

    return consensusMatrix


def generateUniformGrid(originalPoints, gridPoints, returnNeighbors=False):
    """
    Generate a uniform grid of points to cover a region described by a (likely
    non-uniform) set of points.

    Parameters
    ----------

    originalPoints : list(numpy.ndarray)
        Several sets of points describing the region, which are used to find the extreme
        values that need to be included.

    gridPoints : int
        The number of grid points in each direction for the new grid

    returnNeighbors : bool
        Whether or not to calculate the nearest neighbors of the points as well. Not
        recommended to use, since this can be done much faster by findNodeNeighbors using
        a kd-tree.

    Returns
    -------

    numpy.ndarray : The list of points of shape (gridPoints^2, 2)

    or

    (numpy.ndarray, numpy.ndarray) : The list of points of shape (gridPoints^2, 2) and the
        list of nearest neighbor positions for each point.

    """
    # Not exactly the most efficient method to find the extreme points across
    # all configurations, but it works well enough
    flattenedX = []
    flattenedY = []

    for i in range(len(originalPoints)):
        for j in range(len(originalPoints[i])):
            flattenedX.append(originalPoints[i][j,0])
            flattenedY.append(originalPoints[i][j,1])

    uniformX = np.linspace(min(flattenedX), max(flattenedX), gridPoints)
    uniformY = np.linspace(min(flattenedY), max(flattenedY), gridPoints)

    uniformGridPoints = np.zeros([gridPoints, gridPoints, 2])
    neighborDirections = [[1,0], [-1,0], [0,1], [0,-1], [1,1], [1,-1], [-1,1], [-1,-1]]
    gridPointNeighbors = [[] for i in range(gridPoints*gridPoints)]
    
    for y in range(gridPoints):
        for x in range(gridPoints):
            uniformGridPoints[y,x] = [uniformX[x], uniformY[y]]
            for nndir in neighborDirections:
                if ((x+nndir[0] >= 0 and x+nndir[0] < len(uniformX)) and (y+nndir[1] >= 0 and y+nndir[1] < len(uniformY))):
                    gridPointNeighbors[y*gridPoints+x].append(np.array([uniformX[x+nndir[0]], uniformY[y+nndir[1]]]))

    uniformGridPoints = uniformGridPoints.reshape([gridPoints*gridPoints, 2])

    if not returnNeighbors:
        return uniformGridPoints

    return uniformGridPoints, gridPointNeighbors


def consensusCluster(pointArr, communityArr, consensusMatrixCalculationMethod="exact", fastMatrixNumNeighbors=8, fastMatrixMonteCarloStepFactor=1, numCommunityDetections=5, numHiddenDetections=3, gridPoints=60, resolutionCenter=.75, resolutionVar=.05, figOutputFolder=None, showIntermediateResults=False):
    """
    A full pipeline method to go from a set of input community detections to a final consistent
    detection. The method goes through the following steps:

    1. Create a uniform set of points to describe each region (`gridPoints` x `gridPoints`).

    2. Ensure each detection is described by the exact same set of points by removing unique entries

    3. Calculate the consensus matrix (exactly or approximately) for all of the detections

    4. Perform community detection on the consensus matrix with randomly sampled louvain parameters (centered around
    (`resolutionCenter` with variance `resolutionVar`)

    5. Repeat steps 3 and 4 several (`numHiddenDetections`) times

    6. Perform community detection one final time and return

    Parameters
    ----------

    pointArr : list(numpy.ndarray)
        The array of points that describes the region for each detection. Each detection does not need
        to have the same number/location of points, though the length of pointArr[i,:] should match
        communityArr[i,:].

    communityArr : list(numpy.ndarray) 
        The array of community assignments for each point in each community detections. Each detection does not need
        to have the same number/location of points, though the length of pointArr[i,:] should match
        communityArr[i,:].
    
    consensusMatrixCalculationMethod : str
        How to calculate the consensus matrix for the ensemble of community detections. Options are:

        'exact' : Calculate every element of the consensus matrix as the fraction of detections in which
            a given two points are in the same community. O(n^2) complexity.

        'fast' : Calculate only the nearest neighbors elements of each point as the fraction of detections
            in which a given two points are in the same community. Randomly sample points to fill in
            non-nearest neighbor edge weights, as described in:

        Tandon, A., Albeshri, A., Thayananthan, V., Alhalabi, W., & Fortunato, S. (2019). Fast consensus clustering in complex networks. Physical Review E, 99(4), 042301. [https://doi.org/10.1103/PhysRevE.99.042301](https://doi.org/10.1103/PhysRevE.99.042301)

        Approximately O(n) complexity.

    fastMatrixNumNeighbors : int
        The number of neighbors to exactly calculate the consensus matrix edges weights for when
        using the fast matrix calculation.

    fastMatrixMonteCarloStepFactor : int
        The factor by which to multiply the number of points by to get the total number of Monte
        Carlo steps when using the fast matrix calculation. That is, the total number of sampled
        points (other than nearest neighbors) will be the number of points times this value.

    numComunityDetections : int
        The number of community detections to generate from each consensus matrix.

    numHiddenDetections : int
        The number of times to calculate the consensus matrix and then perform community detection.
        Higher number will give the system more time to come to an agreement on the final detection.
        Exact consensus matrix calculation converges faster than fast calculation (as expected).

    gridPoints : int
        The number of grid points in each direction when creating a uniform grid across the region.
        Actual number of points will end up being less than square of this value, since some points
        will be outside the region.

    resolutionCenter : float
        The center of the sampling distribution for generating values of the louvain resolution
        parameter. Should be between .2 and 2 (approximately).

    resoltionVar : float
        The variance of the sampling distribution for generating values of the louvain resolution
        parameter. Higher variance may require higher numHiddenDetections to converge.

    figOutputFolder : str
        The output folder to save the figures for the initial detections, the hidden detections,
        and the final result. If None, no figures will be saved.

    showIntermediateResults : bool
        Whether to show the initial and hidden detections alongside the final detection (True) or
        just show the final detection (False).


    Returns
    -------

    (numpy.ndarray, numpy.ndarray) : The final set of points describing the region (1st element) and the
        community assignments for each point (2nd element).
"""
    # If we want to save the figures, make sure the directory exists
    if figOutputFolder != None:
        if not os.path.isdir(figOutputFolder):
            os.mkdir(figOutputFolder)

    # Determine what consensus matrix calculation method we are using
    cmMethods = ["exact", "fast"]
    if not consensusMatrixCalculationMethod in cmMethods:
        raise Exception(f'Error: unknown consensus matrix calculation passed (\"{consensusMatrixCalculationMethod}\"; options are {cmMethods}')

    # Calculate a uniform set of grid points that covers the entire region
    uniformGridPoints = generateUniformGrid(pointArr, gridPoints)

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
      
        # Assign the new points to their communities
        # I used to find the nearest neighbors here too, but it is actually much much faster to
        # use a kd tree to search for neighbors, which can be done later
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
        #print(pointCounts)
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

    # All of the points should now be the same, so we can just take the first element of the array
    uniformPoints = uniformPointArr[0]

    # Now we calculate the neighbor indices if we need them
    if consensusMatrixCalculationMethod == 'fast':
        uniformNeighborIndices = findNodeNeighbors(uniformPoints, fastMatrixNumNeighbors)

    # Some debug stuff to make sure that the point reduction/neighbor finding worked
    #pointCounts = [len(uniformPointArr[i]) for i in range(len(uniformPointArr))]
    #communityCounts = [len(uniformCommunityArr[i]) for i in range(len(uniformCommunityArr))]
    #print(pointCounts)
    #print(len(uniformNeighborIndices))
    #print(np.unique([len(uni) for uni in uniformNeighborIndices]))
    #print(communityCounts)

    if figOutputFolder != None and showIntermediateResults:
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
        if consensusMatrixCalculationMethod == "exact":
            consensusMatrix = calculateConsensusMatrix(uniformCommunityArr)
        if consensusMatrixCalculationMethod == "fast":
            consensusMatrix = calculateFastConsensusMatrix(uniformCommunityArr, uniformNeighborIndices, m=fastMatrixMonteCarloStepFactor*len(uniformPoints))


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

        if figOutputFolder != None and showIntermediateResults:
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
    if consensusMatrixCalculationMethod == "exact":
        consensusMatrix = calculateConsensusMatrix(uniformCommunityArr)
    if consensusMatrixCalculationMethod == "fast":
        consensusMatrix = calculateFastConsensusMatrix(uniformCommunityArr, uniformNeighborIndices, m=fastMatrixMonteCarloStepFactor*len(uniformPoints))
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
