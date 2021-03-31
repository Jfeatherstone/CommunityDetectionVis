from scipy.spatial import Voronoi, voronoi_plot_2d

import alphashape as ap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["figure.dpi"] = 150

import pandas as pd

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

    # The denominator should be zero for parallel lines
    if denominator == 0:
        return None

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
   
    # Format our line as [[x1, x2], [y1, y2]]
    extendedLine = np.array([[point[0], point[0]], [point[1], highestHullPoint]])
    
    intersectionCount = numLineIntersections(hullLines, extendedLine)
   
    # If a line intersects with the boundary of a shape an even number of times
    # in total, it must be outside that shape (think Qext for Gauss's Law)
    if intersectionCount % 2 == 0:
        return False

    # Otherwise it is inside
    return True

# This method allows us to get rid of excess lines in the picture
def cullVoronoi(vor: Voronoi, communities, hullLines):
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
            # (scipy's library will give -1 as the index of the vertex if it extends to infinity)
            if v[0] == -1 or v[1] == -1:
                pass
            else:
                v1 = vor.vertices[v[0]]
                v2 = vor.vertices[v[1]]

                # Make sure that the vertex is actually inside the shape, and not really far away
                if not isInsideHull(v1, hullLines) or not isInsideHull(v2, hullLines):
                    pass
                else:
                    # Line format: [[x1, x2], [y1, y2]]
                    goodLines.append([[v1[0], v2[0]], [v1[1], v2[1]]])

    return np.array(goodLines)


# Some methods to help visualize things
def drawPoints(ax, pointArr, communities, colors, s=10, noaxis=True):
#    for i in range(len(pointArr)):
#        ax.scatter(pointArr[i,0], pointArr[i,1],
#                   color=colors[communities[i]], s=s)
    # We can do this a bit more efficiently by coloring all of the points in a given community
    # at once
    for comIndex in range(max(communities)+1):
        pointsInCommunity = pointArr[communities == comIndex]
        ax.scatter(pointsInCommunity[:,0], pointsInCommunity[:,1], color=colors[comIndex], s=s)

    if noaxis:
        ax.set_yticks([])
        ax.set_xticks([])

def drawLines(ax, lines, color='black', opacity=1):
    for i in range(len(lines)):
        ax.plot(lines[i,0], lines[i,1], c=color, alpha=opacity)

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def genRandomColors(size, seed=21):
    np.random.seed(seed)

    randomColors = [f"#{rgb_to_hex(tuple(np.random.choice(range(256), size=3).flatten()))}" for i in range(size)]

    return randomColors

def mapPointToColor(point):
    if len(np.shape(point)) == 1:
        # We'll vary red across x, green across y, and blue across both
        # Turns out the trig approach doesn't really work, so we'll just
        # use a (mostly) linear gradient
        #red = int(np.cos(frequencyX*point[0])*127) + 128
        #green = int(np.cos(frequencyY*point[1])*127) + 128
        #blue = int(np.sin(frequencyX*point[0] + frequencyY*point[1])*127) + 128
        # These are just the approximate boundaries of our region
        #extremaX = [640300, 643500]
        #extremaY = [3969000, 3972000]
        #r = int((point[0] - extremaX[0]) / (extremaX[1] - extremaX[0]) * 255)
        #g = int((point[1] - extremaY[0]) / (extremaY[1] - extremaY[0]) * 255)
        #b = int((point[0] % 1000 ) * 128/1000 + (point[1] % 1000 ) * 128/1000)
        seed = int(np.sqrt(point[0]*point[1]/3))
        #print(seed)
        np.random.seed(seed)
        r, g, b = np.random.randint(0, 255, size=3)
        return f'#{rgb_to_hex(tuple([r,g,b]))}'

    # Otherwise recurse to retrieve list of colors
    colors = []
    for i in range(len(point)):
        colors.append(mapPointToColor(point[i]))

    return colors
    
