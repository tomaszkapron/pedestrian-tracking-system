from src.utils import PicBBox


class TrackedROI:
    def __init__(self, objectId: int):
        self.objectId = objectId
        self.age = 0

    def getObjectId(self):
        return self.objectId

    def getAge(self):
        return self.age


class ROITracker:

    def __init__(self, firstFrame: PicBBox, featureTracker):
        self.FT = featureTracker
        self.nextFrame = None
        self.currentFrame = firstFrame
        self.trackedObjects = dict()
        self.history = []
        self.processFirstFrame()

    def processFirstFrame(self):
        rois = self.currentFrame.getROIS()
        for count, roi in enumerate(rois):
            self.trackedObjects[count] = TrackedROI(count)

    def update(self, nextFrame: PicBBox):
        self.nextFrame = nextFrame

        currentROIs = self.currentFrame.getROIS()
        nextROIs = self.nextFrame.getROIS()

        matches = self.matchROIs(currentROIs, nextROIs)

        print(0)

    def matchROIs(self, currentROIs: list, nextROIs: list) -> dict:
        matches = dict()
        for count, currRoi in enumerate(currentROIs):
            matchesList = []
            for nextRoi in currentROIs:
                result = self.FT.featureMatchVis(currRoi, nextRoi, vis=True)
                matchesList.append(result)
            matches[count] = matchesList

        return matches