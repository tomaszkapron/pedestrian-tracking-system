import cv2
import numpy as np
from typing import List


class BBox:
    def __init__(self, textData: str):
        data = textData.split()
        if len(data) != 4:
            raise IndexError("Wrong number of arguments!")
        self.x = int(float(data[0]))
        self.y = int(float(data[1]))
        self.width = int(float(data[2]))
        self.height = int(float(data[3]))

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height


class PicBBox:
    def __init__(self, frame: np.ndarray, bboxList: List[BBox]):
        self.frame = frame
        self.bboxList = bboxList

    def printPicBBox(self):
        cv2.namedWindow("frame")
        for bbox in self.bboxList:
            cv2.rectangle(self.frame,
                          (bbox.getX(), bbox.getY()),
                          (bbox.getX() + bbox.getWidth(), bbox.getY() + bbox.getHeight()),
                          (255, 0, 0), 2)

        cv2.imshow("frame", self.frame)
        cv2.waitKey(0)
        cv2.destroyWindow("frame")

    def printTwoPics(self, second):
        cv2.namedWindow("TwoFrames")
        # concatenate image Vertically
        img = np.concatenate((self.frame, second), axis=0)

        scale_percent = 40  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('TwoFrames', resized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getFirstROI(self):
        x1 = self.bboxList[0].getX()
        y1 = self.bboxList[0].getY()
        w1 = self.bboxList[0].getWidth()
        h1 = self.bboxList[0].getHeight()
        ROI = self.frame[y1:y1 + h1, x1:x1 + w1]
        return ROI

    def getSecondROI(self):
        x1 = self.bboxList[1].getX()
        y1 = self.bboxList[1].getY()
        w1 = self.bboxList[1].getWidth()
        h1 = self.bboxList[1].getHeight()
        ROI = self.frame[y1:y1 + h1, x1:x1 + w1]
        return ROI

    def getROIS(self) -> List[np.ndarray]:
        listOfROIs = []
        for bbox in self.bboxList:
            x1 = bbox.getX()
            y1 = bbox.getY()
            w1 = bbox.getWidth()
            h1 = bbox.getHeight()
            ROI = self.frame[y1:y1 + h1, x1:x1 + w1]
            listOfROIs.append(ROI)
        return listOfROIs

    def getFrame(self):
        return self.frame

    def getNumberOfObjs(self):
        return len(self.bboxList)
