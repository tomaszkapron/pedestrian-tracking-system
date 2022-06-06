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
    def __init__(self, firstFrame: PicBBox):
        self.lastFrame = None
        self.newFrame = firstFrame
        self.trackedObjects = dict()
