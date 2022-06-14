import copy
from typing import Any

from src.utils import PicBBox
from src.utils.utils import getRepeats
from src.utils.TrackedROI import TrackedROI


class ROITracker:
    """Class for tracking pedestrians with Bayes Graph Model using feature matching and histogram matching

    Main method of this class is update(). It is called for every two next frames. Class keep track of every pedestrian
    between frames.

    Attributes:
        FM             - instance of feature matching class
        HM             - instance of histogram matching class
        GM             - instance of graph model
        currentFrame   - PicBox instance of current frame
        nextFrame      - PicBox instance of next frame after current frame
        trackedObjects - dict; key - id of ROI(pedestrian) on current frame, value instance of TrackedROI

    """

    def __init__(self, featureMatcher, histMatcher, graphModel):
        self.FM = featureMatcher
        self.HM = histMatcher
        self.GM = graphModel

        self.currentFrame = None
        self.nextFrame = None

        self.trackedObjects = dict()

    def processFirstFrame(self, firstFrame: PicBBox):
        self.nextFrame = firstFrame
        rois = self.nextFrame.getROIS()
        for count, roi in enumerate(rois):
            self.trackedObjects[count] = TrackedROI(count)

    def update(self, nextFrame: PicBBox):
        # First pass - there is nothing to compare
        if self.nextFrame is None:
            self.processFirstFrame(nextFrame)
            return

        self.printTrackedObj()  # printing data according to project rules
        self.currentFrame = self.nextFrame
        self.nextFrame = nextFrame

        currentROIs = self.currentFrame.getROIS()
        nextROIs = self.nextFrame.getROIS()

        featureMatches, histMatches = self.processROIs(currentROIs, nextROIs)
        normFeature, normHist = self.normalizeForGraph(copy.deepcopy(featureMatches), copy.deepcopy(histMatches))
        self.deductingWithGraph(normFeature, normHist)
        # self.currentFrame.printTwoPics(self.nextFrame.frame)

    def processROIs(self, currentROIs: list, nextROIs: list) -> (dict, dict):
        """
        method for getting features matches and histogram matches for every pair of ROIs between two next frames

        :param currentROIs: list of all ROIs(every pedestian bounding box) on current frame
        :param nextROIs: list of all ROIs(every pedestian bounding box) on next frame
        :return: dict for features matches, and histogram matches; key - ROI id on current frame, value - list of values
         of features for every ROI on next frame
        """
        featureMatches = dict()
        histMatches = dict()
        for count, currRoi in enumerate(currentROIs):
            # print("_____________________________________________________________________________")
            # print(f"\nprocessing {count} bbox, with has id: {self.trackedObjects[count].objectId}")
            # print()
            featureMatchesList = []
            histMatchesList = []

            for nextRoi in nextROIs:
                featMatchNum = self.FM.featureMatchVis(currRoi, nextRoi, vis=False)
                histResult = self.HM.histMatch(currRoi, nextRoi)

                featureMatchesList.append(featMatchNum)
                histMatchesList.append(histResult)

            featureMatches[count] = featureMatchesList
            histMatches[count] = histMatchesList

        return featureMatches, histMatches

    def deductingWithGraph(self, featMatches: dict, histMatches: dict):
        # Fill probability dict with match probabilities, based on features values using Bayes Model
        probabilityDict = dict()
        for key in list(featMatches.keys()):
            featList = featMatches[key]
            histList = histMatches[key]

            probabilityDict[key] = [self.GM.query(featList[i], histList[i]) for i in range(len(featList))]

        # Deducting matches between ROIs
        newTrackedObjs = self.matchROIsFromProbabilityDict(probabilityDict)

        # Logic for finding free id's for new ROIs
        idListForNewTrackedObjs = self.getListOfTrackedIds(newTrackedObjs)
        numberOfNewIds = self.nextFrame.getNumberOfObjs() - len(idListForNewTrackedObjs)

        availableIds = self.getListOfFreeIds(numberOfNewIds, idListForNewTrackedObjs)

        # Adding new tracked obj that just got on the frame. Initialization with free id's
        for i in range(self.nextFrame.getNumberOfObjs()):
            if i in newTrackedObjs:
                continue
            newTrackedObjs[i] = TrackedROI(availableIds.pop())

        self.trackedObjects = newTrackedObjs

    def matchROIsFromProbabilityDict(self, probabilityDict: dict) -> dict:
        """
        method follows the algorithm:
        1) for every probabilityDict item find the highest probability and save it with its list index,as a set, inside
           the new dict supportMatchesDict
        2) Delete every entry with the highest probability lower than 0.3
        3) If all the list indexes in alle sets are unique stop the algorithm - it means that's there is trivial case,
           and there are no more than 1 matches to 1 ROI on next frame
        4) Change the lowest probability multimatch to next highest probality and update supportMatchesDict with new
           value
        5) Goto step 2

        :param probabilityDict: dict where, key - ROI id for current frame, value - list of probabilities for being the
               same pedestrian, between ROI with key id, and every ROI on next frame
        :return: dict with matching frames between two frames where key - index of ROI for current frame,
               value - TrackedROI
        """
        newTrackedObjs = dict()
        supportMatchesDict = dict()
        for boxNum, probList in probabilityDict.items():
            # find best match and save it as best match (bestProbability, objIdFromNextFrame)
            bestMatch = (-1, -1)
            for objId, prob in enumerate(probList):
                if prob >= bestMatch[0]:
                    bestMatch = (prob, objId)

            supportMatchesDict[boxNum] = [probList, bestMatch]

        deductedMatches = False
        while not deductedMatches:
            supportMatchesDict = {key: val for key, val in supportMatchesDict.items() if val[1][0] > 0.3}
            objIds = [val[1][1] for key, val in supportMatchesDict.items()]
            # Compare length for unique elements
            if len(set(objIds)) == len(objIds):
                deductedMatches = True
                break
            repeated = getRepeats(objIds)
            lowestProb = (1.1, -1)
            for boxNum, val in supportMatchesDict.items():
                if val[1][1] == repeated[0] and val[1][0] < lowestProb[0]:
                    lowestProb = (val[1][0], boxNum)

            probLowList = supportMatchesDict[lowestProb[1]][0]

            newBestMatch = (-1, -1)
            for objId, prob in enumerate(probLowList):
                if newBestMatch[0] <= prob < supportMatchesDict[lowestProb[1]][1][0]:
                    newBestMatch = (prob, objId)

            supportMatchesDict[lowestProb[1]] = [supportMatchesDict[lowestProb[1]][0], newBestMatch]

        for boxNum, val in supportMatchesDict.items():
            trackedObj = self.trackedObjects[boxNum]
            newTrackedObjs[val[1][1]] = trackedObj.incrementAge()

        return newTrackedObjs

    @staticmethod
    def normalizeForGraph(featMatches: dict, histMatches: dict) -> (dict, dict):
        """
        :param featMatches:
        :param histMatches:
        :return: dicts with normalized values for graph model

        for number of feature matches:
        0 - 3 -> 0
        4 - 6 -> 1
        6 - 9 -> 2
        >9    -> 3

        for hist matches:
        0 - 0.3    -> 0
        0.3 - 0.4  -> 1
        0.4 - 0.5  -> 2
        0.5 - 0.6  -> 3
        0.6 - 0.7  -> 4
        0.7 - 0.8  -> 5
        >0.8       -> 6
        """

        def transformFeatures(val: int) -> int:
            if val <= 3:
                return 0
            if val <= 6:
                return 1
            if val <= 9:
                return 2
            return 3

        def transformHist(val: int) -> int:
            if val <= 0.3:
                return 0
            if val <= 0.4:
                return 1
            if val <= 0.5:
                return 2
            if val <= 0.6:
                return 3
            if val <= 0.7:
                return 4
            return 5

        for boxNum, matchList in featMatches.items():
            featMatches[boxNum] = [transformFeatures(val) for val in matchList]

        for boxNum, histList in histMatches.items():
            histMatches[boxNum] = [transformHist(val) for val in histList]

        return featMatches, histMatches

    @staticmethod
    def getListOfTrackedIds(trackedObjDict) -> list:
        listOfTrackedIds = []
        for frameId, obj in trackedObjDict.items():
            listOfTrackedIds.append(obj.objectId)

        return listOfTrackedIds

    @staticmethod
    def getListOfFreeIds(numberOfIdsNeeded: int, takenIds: list[int]) -> list[int]:
        freeIds = []
        suspectId = 0
        while len(freeIds) != numberOfIdsNeeded:
            if suspectId in takenIds:
                suspectId += 1
                continue
            freeIds.append(suspectId)
            suspectId += 1

        return freeIds

    def printTrackedObj(self):
        resultStr = ""
        for i, trackedObj in self.trackedObjects.items():
            resultStr += str(trackedObj)
            resultStr += " "
        print(resultStr)
