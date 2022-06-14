class TrackedROI:
    """
    class for keeping information about single object(pedestrian) on the scene. When pedestrain get in the frame
    instance of the class is created, and beeing passed between frames until pedestrian leave the frame

    Attributes:
        objectId       - unique ID for the pedestrian
        age            - indicates for how many frames object is on the scene
        isNew          - inforation if object was just created
    """
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