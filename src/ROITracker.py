import copy
from typing import Any

from src.utils import PicBBox


class TrackedROI:
    def __init__(self, objectId: int):
        self.objectId = objectId
        self.age = 0
        self.isNew = True

    def getObjectId(self):
        return self.objectId

    def getAge(self):
        return self.age

    def incrementAge(self):
        self.age += 1
        self.isNew = False
        return self

    def __str__(self):
        if self.isNew:
            return '-1'
        else:
            return str(self.objectId)


class ROITracker:

    def __init__(self, featureMatcher, histMatcher, graphModel):
        self.FM = featureMatcher
        self.HM = histMatcher
        self.GM = graphModel

        self.currentFrame = None
        self.nextFrame = None

        self.trackedObjects = dict()
        self.history = []

    def processFirstFrame(self, firstFrame: PicBBox):
        self.nextFrame = firstFrame
        rois = self.nextFrame.getROIS()
        for count, roi in enumerate(rois):
            self.trackedObjects[count] = TrackedROI(count)

    def update(self, nextFrame: PicBBox):
        # First pass -there is nothing to compare
        if self.nextFrame is None:
            self.processFirstFrame(nextFrame)
            return

        self.printTrackedObj()
        self.currentFrame = self.nextFrame
        self.nextFrame = nextFrame

        currentROIs = self.currentFrame.getROIS()
        nextROIs = self.nextFrame.getROIS()

        featureMatches, histMatches = self.processROIs(currentROIs, nextROIs)
        normFeature, normHist = self.normalizeForGraph(copy.deepcopy(featureMatches), copy.deepcopy(histMatches))
        self.deductingWithGraph(normFeature, normHist)
        # print(prob)
        # self.currentFrame.printTwoPics(self.nextFrame.frame)

    def processROIs(self, currentROIs: list, nextROIs: list) -> (dict, dict):
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
        probabilityDict = dict()

        for key in list(featMatches.keys()):
            featList = featMatches[key]
            histList = histMatches[key]

            probabilityDict[key] = [self.GM.query(featList[i], histList[i]) for i in range(len(featList))]

        self.history.append(copy.deepcopy(self.trackedObjects))
        newTrackedObjs = dict()

        for boxNum, matchList in probabilityDict.items():
            # find best match and save it as
            bestMatch = (-1, -1)
            for objId, matchesNum in enumerate(matchList):
                if matchesNum >= bestMatch[0]:
                    bestMatch = (matchesNum, objId)

            if bestMatch[0] > 0.3:
                trackedObj = self.trackedObjects[boxNum]
                newTrackedObjs[bestMatch[1]] = trackedObj.incrementAge()
            # else: # it probably means that roi is out of frame

        idListForNewTrackedObjs = self.getListOfTrackedIds(newTrackedObjs)
        numberOfNewIds = self.nextFrame.getNumberOfObjs() - len(idListForNewTrackedObjs)

        availableIds = self.getListOfFreeIds(numberOfNewIds, idListForNewTrackedObjs)

        # adding new tracked obj that just got on the frame
        for i in range(self.nextFrame.getNumberOfObjs()):
            if i in newTrackedObjs:
                continue
            newTrackedObjs[i] = TrackedROI(availableIds.pop())

        self.trackedObjects = newTrackedObjs

    def normalizeForGraph(self, featMatches: dict, histMatches: dict) -> (dict, dict):
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

    # TODO: this func will be replaced with graph model
    def deducting(self, featureMatches: dict):
        self.history.append(copy.deepcopy(self.trackedObjects))
        newTrackedObjs = dict()

        for boxNum, matchList in featureMatches.items():
            # find best match and save it as
            bestMatch = (-1, -1)
            for objId, matchesNum in enumerate(matchList):
                if matchesNum >= bestMatch[0]:
                    bestMatch = (matchesNum, objId)

            # TODO: issue: same matches value
            if bestMatch[0] > 0:
                trackedObj = self.trackedObjects[boxNum]
                newTrackedObjs[bestMatch[1]] = trackedObj.incrementAge()
            # else: # it probably means that roi is out of frame

        idListForNewTrackedObjs = self.getListOfTrackedIds(newTrackedObjs)
        numberOfNewIds = self.nextFrame.getNumberOfObjs() - len(idListForNewTrackedObjs)

        availableIds = self.getListOfFreeIds(numberOfNewIds, idListForNewTrackedObjs)

        # adding new tracked obj that just got on the frame
        for i in range(self.nextFrame.getNumberOfObjs()):
            if i in newTrackedObjs:
                continue
            newTrackedObjs[i] = TrackedROI(availableIds.pop())

        self.trackedObjects = newTrackedObjs

    def getTrackedObjectFromTrackedObj(self, objectId) -> Any | None:
        for frameId, obj in self.trackedObjects.items():
            if obj.objectId == objectId:
                return obj

        return None

    def getListOfTrackedIds(self, trackedObjDict) -> list:
        listOfTrackedIds = []
        for frameId, obj in trackedObjDict.items():
            listOfTrackedIds.append(obj.objectId)

        return listOfTrackedIds

    def getListOfFreeIds(self, numberOfIdsNeeded: int, takenIds: list[int]) -> list[int]:
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
