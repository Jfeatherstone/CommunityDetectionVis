"""
A set of tools to analyze a set of community detections performed on a (possibly heterogeneous) graph, primarily using consensus clustering.

Principle method is ConsensusClustering.consensusCluster(), which takes in a set of input community detections, and outputs a single detection computed according to the method defined in:

Lancichinetti, A., & Fortunato, S. (2012). Consensus clustering in complex networks. Scientific Reports, 2(1), 336. [https://doi.org/10.1038/srep00336](https://doi.org/10.1038/srep00336)

For very high density graphs, clustering can be done using the 'fast' method as described in this paper:

Tandon, A., Albeshri, A., Thayananthan, V., Alhalabi, W., & Fortunato, S. (2019). Fast consensus clustering in complex networks. Physical Review E, 99(4), 042301. [https://doi.org/10.1103/PhysRevE.99.042301](https://doi.org/10.1103/PhysRevE.99.042301)

"""

from .Visualization import *
from .CommunityWise import *

from .Utils import *
