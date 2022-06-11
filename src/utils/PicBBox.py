import cv2
import numpy as np

from src.utils import BBox


class PicBBox:
    def __init__(self, frame: np.ndarray, bboxList: list[BBox]):
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

    def getROIS(self) -> list[np.ndarray]:
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
