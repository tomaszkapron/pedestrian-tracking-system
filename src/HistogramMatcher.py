import cv2
import numpy as np


class HistMatcher:
    def __init__(self):
        self.h_bins = 50
        self.s_bins = 60
        self.histSize = [self.h_bins, self.s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        self.h_ranges = [0, 180]
        self.s_ranges = [0, 256]
        self.ranges = self.h_ranges + self.s_ranges  # concat lists
        # Use the 0-th and 1-st channels
        self.channels = [0, 1]

    def histMatch(self, oldROI: np.ndarray, newROI: np.ndarray) -> int:
        """
        :param oldROI
        :param newROI

        :return int value describing how similar two rois are
        """
        hsv_old = cv2.cvtColor(oldROI, cv2.COLOR_BGR2HSV)
        hsv_new = cv2.cvtColor(newROI, cv2.COLOR_BGR2HSV)

        hist_old = cv2.calcHist([hsv_old], self.channels, None, self.histSize, self.ranges, accumulate=False)
        cv2.normalize(hist_old, hist_old, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_new = cv2.calcHist([hsv_new], self.channels, None, self.histSize, self.ranges, accumulate=False)
        cv2.normalize(hist_new, hist_new, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return cv2.compareHist(hist_old, hist_new, 0)
