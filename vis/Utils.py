import pandas as pd
import os

import numpy as np

from .CommunityWise import patchCommunityIndices

def loadCommunityData(gridFolder, communityFolder, goodParameterValues=['0.70', '0.80', '0.90']):
    # Read in the different communities we have
    communityFiles = os.listdir(communityFolder)
    gridFiles = os.listdir(gridFolder)

    # Note the annoying usual files so they don't mess things up
    badFiles = ['.', '..', '.ipynb_checkpoints', '__pycache__']

    communityArr = []
    pointArr = []

    # We iterate over the grid files, since there may be multiple detections for a single
    # grid (with different parameter values)
    for file in gridFiles:
        if file in badFiles:
            continue

        # Find the number that this grid corresponds to (should be between 1 and 20)
        fileIndex = file.replace('.csv', '').replace('DM', '')

        # Read in the positions of the nodes
        gridDataFrame = pd.read_csv(f'{gridFolder}/{file}', sep=" ")
        currPoints = np.zeros([len(gridDataFrame["x"]), 2])
        currPoints[:,0] = gridDataFrame["x"]
        currPoints[:,1] = gridDataFrame["y"]

        # For a given index $i, the community files will be called 'DMiGX.XX.csv'
        # where the X's represent the parameter value
        # Now we can patch together the file names of the good files, and try to
        # find them
        if goodParameterValues == None:
            goodCommunityFileNames = [communityFiles[i] for i in range(len(communityFiles)) if f'DM{fileIndex}G' in communityFiles[i]]
        else:
            goodCommunityFileNames = [f'DM{fileIndex}G{goodParameterValues[i]}.csv' for i in range(len(goodParameterValues))]

        # Make sure that these files actually exist, and if they do, we save their data
        for goodFile in goodCommunityFileNames:
            if goodFile in communityFiles:
                currComm = np.genfromtxt(f'{communityFolder}/{goodFile}', dtype=int)

                # Make sure there aren't any discontinuities in the community indices
                currComm = patchCommunityIndices(currComm)
                # Make sure that things line up (and we have actual community info)
                # and then add to the running list
                if np.sum(currComm) != 0 and len(currComm) == len(currPoints):
                    communityArr.append(currComm)
                    pointArr.append(currPoints)
                else:
                    print(f'Misalignment or invalid communities: {goodFile}')

            else:
                print(f'Missing expected file: {goodFile}')

    return pointArr, communityArr

