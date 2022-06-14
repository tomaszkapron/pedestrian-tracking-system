import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureMatcher:
    def __init__(self):
        self.MIN_MATCH_COUNT = 4
        self.FLANN_INDEX_KDTREE = 2
        self.swift = cv2.SIFT_create()

    def featureMatchVis(self, oldROI: np.ndarray, newROI: np.ndarray, vis=True) -> int:
        """
        :param vis -> if set to True visualising matches
        :param oldROI
        :param newROI

        :return int value of good matches between two ROIs
        """
        # convert to grayscale
        newROI = cv2.cvtColor(newROI, cv2.COLOR_BGR2GRAY)
        oldROI = cv2.cvtColor(oldROI, cv2.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.swift.detectAndCompute(newROI, None)
        kp2, des2 = self.swift.detectAndCompute(oldROI, None)

        index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=10)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            # print("Enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
            if M is None:
                M = np.float32([[1.05764315e-02, -2.96058593e-01, 4.90867768e+01],
                                [-1.58998786e-01, -2.86754871e-01, 6.31794145e+01],
                                [-8.05060010e-04, -5.48503251e-03, 1.00000000e+00]])
            matchesMask = mask.ravel().tolist()
            h, w = newROI.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2 = cv2.polylines(oldROI, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            if vis:
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                img3 = cv2.drawMatches(newROI, kp1, oldROI, kp2, good, None, **draw_params)
                plt.imshow(img3, 'gray'), plt.show()

            return len(good)

        else:
            # print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            if vis:
                img3 = cv2.drawMatches(newROI, kp1, oldROI, kp2, good, None)
                plt.imshow(img3, 'gray'), plt.show()
            matchesMask = None
            return 0

