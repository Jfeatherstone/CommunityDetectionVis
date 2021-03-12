from scipy.spatial import Voronoi, voronoi_plot_2d

import alphashape as ap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["figure.dpi"] = 150

import pandas as pd
from colour import Color

# Calculates the intersection point between two finite lines (or None)
# algorithm from: jk I used this algorithm in a previous project and
# the link that I had there no longer works, so....
# Whatever, the algorithm works, so hopefully that's not a big deal
# I think it is just a generic parametric line intersection calculation
def intersection(l1, l2):
    # I have renamed the variables like this to be consistent with the source
    # above (or maybe not)
    l1 = np.array(l1)
    l2 = np.array(l2)

    x = [0, l1[0,0], l1[0,1], l2[0,0], l2[0,1]]
    y = [0, l1[1,0], l1[1,1], l2[1,0], l2[1,1]]
    
    denominator = ((x[4] - x[3])*(y[2] - y[1]) - (x[2] - x[1])*(y[4] - y[3]))
   
    s = ((x[4]-x[3])*(y[3]-y[1]) - (x[3]-x[1])*(y[4] - y[3])) / denominator

    t = ((x[2]-x[1])*(y[3]-y[1]) - (x[3]-x[1])*(y[2] - y[1])) / denominator
    
    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        intersectionPoint = [x[1] + (x[2] - x[1])*s, y[1] + (y[2] - y[1])*s]
        return intersectionPoint
    
    return None

# Computes the number of elements of lineSet that line intersects
def numLineIntersections(lineSet, line):
    count = 0
    for i in range(len(lineSet)):
        if intersection(lineSet[i], line) != None:
            count += 1
    return count
    
# This allows us to check if a point is inside a given hull (aka set of lines)
def isInsideHull(point, hullLines, padding=1000):
    # First, we extend the line up past the edge of the hull
    highestHullPoint = np.max(hullLines[:,1,0]) + padding
    
    extendedLine = np.array([[point[0], point[0]], [point[1], highestHullPoint]])
    
    intersectionCount = numLineIntersections(hullLines, extendedLine)
    
    if intersectionCount % 2 == 0:
        return False
    return True

# This method allows us to get rid of excess lines in the picture
def cullVoronoi(vor: Voronoi, communities, hullLines, boundaryPadding=500):
    
    # If we are provided a cutoff distance, we should get rid of any points that are
    # farther than that from the center of all of our points
    #mapTopLeft = [np.max(vor.points[:,0]), np.max(vor.points[:,1])]
    #mapBottomRight = [np.max(vor.points[:,0]), np.max(vor.points[:,1])]
    
    # We can't know how many lines will be good in advance, so we just
    # start with an empty array and append as we go
    goodLines = []
    
    # We iterate over all of the ridges (lines) in the voronoi tessellation
    for k, v in vor.ridge_dict.items():
        # The key will be a tuple with the indices of the two points this line divides
        # The value will be an array with the indices of the two vertices that define the line
        
        # If they are in the same community, we don't need to draw that line
        # Otherwise, we do
        if communities[k[0]] != communities[k[1]]:
            # We don't care about lines that extend to infinity, so ignore those
            if v[0] == -1 or v[1] == -1:
                pass
            else:
                # Now make sure that these vertices aren't farther than our cutoff distance
                v1 = vor.vertices[v[0]]
                v2 = vor.vertices[v[1]]
                #print(v1)
                #print(np.sqrt((v1[0] - mapCenter[0])**2 + (v1[0] - mapCenter[1])**2))
                #if np.sqrt((v1[0] - mapCenter[0])**2 + (v1[1] - mapCenter[1])**2) > cutoffDistance or np.sqrt((v2[0] - mapCenter[0])**2 + (v2[1] - mapCenter[1])**2) > cutoffDistance:
                if not isInsideHull(v1, hullLines) or not isInsideHull(v2, hullLines):
                    pass
                else:
                    goodLines.append([[v1[0], v2[0]], [v1[1], v2[1]]])

    return np.array(goodLines)


# Some methods to help visualize things
def drawPoints(ax, pointArr, communities, colors):
    for i in range(len(points)):
        ax.scatter(pointArr[i,0], pointArr[i,1], color=colors[communities[i]-1])

def drawLines(ax, lines, color='black', opacity=1):
    for i in range(len(lines)):
        ax.plot(lines[i,0], lines[i,1], c=color, alpha=opacity)

def genRandomColors(size, seed=21):
    np.random.seed(seed)

    def rgb_to_hex(rgb):
        return '%02x%02x%02x' % rgb

    randomColors = [f"#{rgb_to_hex(tuple(np.random.choice(range(256), size=3).flatten()))}" for i in range(size)]

    return randomColors
