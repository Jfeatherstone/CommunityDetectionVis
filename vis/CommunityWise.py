from scipy.spatial import Voronoi, voronoi_plot_2d

import alphashape as ap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["figure.dpi"] = 150

import pandas as pd
from colour import Color

from .Visualization import isInsideHull, intersection 

def concaveHull(points, communities):
    # Find the total boundary around the shape
    # The .80 is to say that we don't want the boundary to be crazy angular
    alpha = 0.80 * ap.optimizealpha(points)
    hull = ap.alphashape(points, alpha)
    hullPoints = hull.exterior.coords.xy

    # And convert to a better format
    hullPoints = [hullPoints[0].tolist(), hullPoints[1].tolist()]

    # Now we also want to know what communities the points involved
    # in the boundary are (for creating closed shapes later on)
    # You can't use a list as a key for a dictionary, so we take the string
    # that represents the list. Not ideal, but it shouldn't cause any real issues
    communitiesByPointPosition = {}
    for i in range(len(points)):
        communitiesByPointPosition[str([points[i,0], points[i,1]])] = communities[i]

    # Transform to looking at the lines of the hull (instead of the points)
    hullLines = np.zeros([len(hullPoints[0]), 2, 2])
    hullLineCommunities = np.zeros([len(hullPoints[0]), 2])

    for i in range(len(hullPoints[0])):
        # Each of these lines are of the form:
        # [[x1, x2], [y1, y2]]
        hullLines[i] = [[hullPoints[0][i], hullPoints[0][(i+1)%len(hullPoints[0])]],
                        [hullPoints[1][i], hullPoints[1][(i+1)%len(hullPoints[0])]]]

        # This line is pretty disgusting, but unfortunately it can't really be avoided
        # because we need to know which communities the boundary lines connect :/
        hullLineCommunities[i] = [communitiesByPointPosition[str([hullPoints[0][i], hullPoints[1][i]])],
                                  communitiesByPointPosition[str([hullPoints[0][(i+1)%len(hullPoints[0])], hullPoints[1][(i+1)%len(hullPoints[0])]])]]

    return hullLines, hullLineCommunities

# This just makes sure that the community labels are continuous
# and nice numbers
def patchCommunityIndices(communities):
    uniqueCommunities = np.unique(communities)
    replaceDict = dict(zip(uniqueCommunities, range(len(uniqueCommunities))))
    communities = np.array([replaceDict[c] for c in communities])

    return communities


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# This method allows us to get rid of excess lines in the picture
# and also keeps track of which communities are bounded by which lines
def vorToCommunityBounds(vor: Voronoi, communities, hullLines, hullLineCommunities):
    # Instead of just keeping track of whether or not a line is good,
    # we have to keep track of what community it provides the boundary for
    # First, patch the community indices just to make sure we don't have
    # any gaps in indices
    communities = patchCommunityIndices(communities)
    communityBounds = [[] for i in range(max(communities)+1)]

    # This dictionary will contain the communities that we need to fix
    # The key will be the community index, and the value will be the line
    # index that intersects with the boundary
    boundaryCommunitiesToFix = {}

    # We iterate over all of the ridges (lines) in the voronoi tessellation
    for k, v in vor.ridge_dict.items():
        # The key will be a tuple with the indices of the two points this line divides
        # The value will be an array with the indices of the two vertices that define the line
        v1 = vor.vertices[v[0]]
        v2 = vor.vertices[v[1]]
        line = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])

        # If they are in the same community, we don't need to draw that line
        # Otherwise, we do
        if communities[k[0]] != communities[k[1]]:

            # Unlike in the cullVoronoi method in Visualization.py, we actually do care
            # about infinite/far away lines in this case, since we need to make sure
            # that all of our community boundaries are closed shapes
            pointsInsideHull = [int(isInsideHull(v1, hullLines)), int(isInsideHull(v2, hullLines))]

            if v[0] == -1 and v[1] == -1:
                continue

            # If either of the points go to infinity, we have to renormalize the infinite
            # one so that it is just a little outside the shape, and then the next
            # algorithm will take care of it
            if v[0] == -1 or v[1] == -1:
                #print("infinite point")

                if v[0] == -1:
                    nonInfinitePoint = v2
                else:
                    nonInfinitePoint = v1

                # We don't care if the infinite point is outside the shape
                if not isInsideHull(nonInfinitePoint, hullLines):
                    continue

                # We take the non-infinite point as well as the midpoint between the two
                # nodes to calculate a slope, and then extend the line using that
                nodeMidpoint = (vor.points[k[0]] + vor.points[k[1]]) / 2
                lineSlope = (nonInfinitePoint[1] - nodeMidpoint[1]) / (nonInfinitePoint[0] - nodeMidpoint[0])
                lineIntercept = nonInfinitePoint[1] - lineSlope*nonInfinitePoint[0]

                # If the midpoint is to the right of the non infinite point, we must be on the
                # right side of the shape. We can handle both sides by using the sign function
                # 10000 should be enough padding to get us outside of the shape
                newExternalPoint = [nonInfinitePoint[0] + np.sign(nodeMidpoint[0] - nonInfinitePoint[0])*1000,
                                    lineSlope*(nonInfinitePoint[0] + np.sign(nodeMidpoint[0] - nonInfinitePoint[0])*1000) + lineIntercept]

                # Reassign things so the next algorithm can work
                v1 = nonInfinitePoint
                v2 = newExternalPoint
                line = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
                pointsInsideHull = [int(isInsideHull(v1, hullLines)), int(isInsideHull(v2, hullLines))]

            # The sum statement is to check if both elements of the list are true
            if not sum(pointsInsideHull) == 2:
                #continue
                # Somehow if both points are outside the hull, move on
                if sum(pointsInsideHull) == 0:
                    continue

                # We first find the closest intersection with the hull
                if pointsInsideHull[0]:
                    vertexCloseToEdge = v1
                else:
                    vertexCloseToEdge = v2

                closestIntersection = np.zeros(2)
                hullLineIndex = None

                for j in range(len(hullLines)):
                    point = intersection(line, hullLines[j])
                    if point != None and dist(point, vertexCloseToEdge) < dist(closestIntersection, vertexCloseToEdge):
                        closestIntersection = point
                        hullLineIndex = j

                # Assuming we found something, add the line between the vertex and the hull intersection
                if sum(closestIntersection) != 0:
                    #print(communities[k[0]])
                    #print(communities[k[1]])
                    line = np.array([[vertexCloseToEdge[0], closestIntersection[0]], [vertexCloseToEdge[1], closestIntersection[1]]])
                    communityBounds[communities[k[0]]].append(line)
                    communityBounds[communities[k[1]]].append(line)

                    # We will likely need to complete the boundary later on (since part
                    # of it will be the outer hull) so we make a note of that
                    # This correction will happen after the main loop

                    # It is also possible that we will have multiple segments that need to be fixed
                    # (imagine a crescent moon shape with the points on the boundary) so
                    # we need a list of all the intersects
                    if communities[k[0]] not in boundaryCommunitiesToFix:
                        boundaryCommunitiesToFix[communities[k[0]]] = []
                    if communities[k[1]] not in boundaryCommunitiesToFix:
                        boundaryCommunitiesToFix[communities[k[1]]] = []

                    boundaryCommunitiesToFix[communities[k[0]]].append([len(communityBounds[communities[k[0]]]) - 1, hullLineIndex])
                    boundaryCommunitiesToFix[communities[k[1]]].append([len(communityBounds[communities[k[1]]]) - 1, hullLineIndex])

            else:
                # And append the line to both of the communities involved
                # Line format: [[x1, x2], [y1, y2]]
                communityBounds[communities[k[0]]].append(line)
                communityBounds[communities[k[1]]].append(line)


    # As promised, we have to complete the boundaries of communities that
    # lie on the edge of our region, since there won't be ridges from
    # the voronoi tesselation
    # To do this, we use the lines from the concave hull itself. This does
    # mean that the area attributed to a node is clipped a little around
    # the boundaries, but since it is outside of the shape, this shoudn't
    # be an issue
    for comIndex, lineIndices in boundaryCommunitiesToFix.items():
        #print(comIndex, lineIndices)

        # In the very niche case that a community utilizes part of a boundary
        # line (but not long enough to encompass a point) we have to keep track of
        # which lines we've added so we don't double count them
        hullLineIndicesAdded = []

        for i in range(len(lineIndices)):
            # We'll just take the first line index (there could be multiple)
            # Unpack the value into the two things it is
            vorLineIndex, hullLineIndex = lineIndices[i]

            # First, we find the hull line that our line intersects
            # This is something that we actually already found, so we just grab it
            hullIntersectionPoint = communityBounds[comIndex][vorLineIndex][:,1]

            # Now we add all of the connecting points from the vor edge set
            # to the hull edge set
            if hullLineCommunities[hullLineIndex][0] == comIndex:
                connectingSegment = np.array([[hullIntersectionPoint[0], hullLines[hullLineIndex,0,0]], [hullIntersectionPoint[1], hullLines[hullLineIndex,1,0]]])
                communityBounds[comIndex].append(connectingSegment)
            elif hullLineCommunities[hullLineIndex][1] == comIndex:
                connectingSegment = np.array([[hullIntersectionPoint[0], hullLines[hullLineIndex,0,1]], [hullIntersectionPoint[1], hullLines[hullLineIndex,1,1]]])
                communityBounds[comIndex].append(connectingSegment)
            else:
                # If this happens, it means that the portion of the community along the boundary
                # is not the full segment, only a small part of it (less than a single line)
                # Because of this, we can just connect the two points we have.

                # To find out which two segments to connect (since there could be multiple)
                # we take the two that intersect the same hull line
                for k in range(len(lineIndices)):
                    if k != i and lineIndices[k][1] == hullLineIndex:
                        secondIntersection = communityBounds[comIndex][lineIndices[k][0]][:,1]
                        connectingSegment = np.array([[hullIntersectionPoint[0], secondIntersection[0]], [hullIntersectionPoint[1], secondIntersection[1]]])

                        # We have to make sure we don't add the same line twice
                        if hullLineIndex not in hullLineIndicesAdded:
                            hullLineIndicesAdded.append(hullLineIndex)
                            communityBounds[comIndex].append(connectingSegment)

                        # There shouldn't be more than one intersection with the same boundary line
                        # so we are safe to break here
                        break




        # Now that we have all of the connecting lines, we can just add the boundary lines
        # where both points are a part of this community
        # Luckily, we don't care about the order of the lines, so that isn't an issue
        for i in range(len(hullLines)):
            if hullLineCommunities[i][0] == comIndex and hullLineCommunities[i][1] == comIndex:
                communityBounds[comIndex].append(np.array(hullLines[i]))

            # Now that we are on the line, we can just traverse
    #         possibleBoundaries = [connectingSegment1]
    #         for i in range(1, len(hullLines)):
    #             # We will offset our index to start at the hull line used earlier,
    #             # and loop back around
    #             currHullLineIndex = (i + hullLineIndex) % len(hullLines)

    #             # We check if this line intersects with
    #         communityBounds[comIndex].append(connectingSegment1)

    # We have to convert each list of lines to a numpy array
    # WITHOUT converting the actual list of arrays to numpy
    # array. That needs to stay a list since each of the individual
    # arrays vary in length
    for i in range(max(communities)+1):
        communityBounds[i] = np.array(communityBounds[i])


    return communityBounds
