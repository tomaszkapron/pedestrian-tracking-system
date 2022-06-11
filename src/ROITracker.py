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


class ROITracker:

    def __init__(self, firstFrame: PicBBox, featureTracker, histMacher):
        self.FT = featureTracker
        self.HM = histMacher
        self.currentFrame = None
        self.nextFrame = firstFrame

        self.trackedObjects = dict()

        self.history = []
        self.processFirstFrame()

    def processFirstFrame(self):
        rois = self.nextFrame.getROIS()
        for count, roi in enumerate(rois):
            self.trackedObjects[count] = TrackedROI(count)

    def update(self, nextFrame: PicBBox):
        self.currentFrame = self.nextFrame
        self.nextFrame = nextFrame

        currentROIs = self.currentFrame.getROIS()
        nextROIs = self.nextFrame.getROIS()

        featureMatches, histMatches = self.matchROIs(currentROIs, nextROIs)
        self.deducting(featureMatches)

    def matchROIs(self, currentROIs: list, nextROIs: list) -> (dict, dict):
        featureMatches = dict()
        histMatches = dict()
        for count, currRoi in enumerate(currentROIs):
            print(f"\nprocessing {count} bbox, with has id: {self.trackedObjects[count].objectId}")

            featureMatchesList = []
            histMatchesList = []

            for nextRoi in nextROIs:
                featMatchNum = self.FT.featureMatchVis(currRoi, nextRoi, vis=False)
                histResult = self.HM.histMatch(currRoi, nextRoi)

                featureMatchesList.append(featMatchNum)
                histMatchesList.append(histResult)

            featureMatches[count] = featureMatchesList
            histMatches[count] = histMatchesList

        return featureMatches, histMatches

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
