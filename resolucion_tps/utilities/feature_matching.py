import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import List, Tuple
from types import NoneType

class FeatureMatching():
    def __init__(self, lowe_ratio: float = 0.8,
                 flann_algorithm: int = 1, flann_trees: int = 5,
                 flann_checks: int = 50, homography_min_match: int = 8):
        self.lowe_ratio = lowe_ratio
        self.flann_algorithm = flann_algorithm
        self.flann_trees = flann_trees
        self.flann_checks = flann_checks
        self.homography_min_match = homography_min_match

    def match(self, template: cv.typing.MatLike, target: cv.typing.MatLike,
              originalRGB: cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Realiza el match entre un template y un target, utilizando SIFT.
        """
        sift = cv.SIFT_create()

        # Se realizan copias para no sobreescribir los inputs
        template_fin = template.copy()
        target_fin = target.copy()
        rgb_fin = originalRGB.copy()

        template_keypoints, template_descriptors = sift.detectAndCompute(template_fin, None)
        target_keypoints, target_descriptors = sift.detectAndCompute(target_fin,None)

        flann, matches = self.__flann__(template_descriptors=template_descriptors,
                                        target_descriptors=target_descriptors)
        
        good = self.__lowe_ratio__(matches=matches)

        M, imgHomografia = self.__homografia__(img1=template_fin, kp1=template_keypoints,
                                               img2=rgb_fin, kp2=target_keypoints,
                                               good=good)

        return imgHomografia

    def __flann__(self, template_descriptors, target_descriptors):
        """
        Implementa Flann Based Matcher sobre los descriptores de un template y un target.
        El algoritmo KNN busca los dos mejores vecinos (k = 2)
        SOURCE: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        """
        index_params = dict(algorithm = self.flann_algorithm, trees = self.flann_trees)
        search_params = dict(checks = self.flann_checks)
        
        flann = cv.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(template_descriptors, target_descriptors, k=2)

        return flann, matches
    
    def __lowe_ratio__(self, matches, sort=False):
        """
        Implementa el test de Lowe (o Lowe's Ratio test) sobre un conjunto de matches.
        Source: https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
        """
        good = []
        for m, n in matches:
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)

        if sort:
            good = sorted(good, key = lambda x:x.distance)
        
        return good
    
    ## Permite aplicar una homografía
    # SOURCE: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
    def __homografia__(self, img1, kp1, img2, kp2, good, color = (22, 168, 53)) -> Tuple[list[float], cv.typing.MatLike]:
        """
        Calcula una homografía en función de un template, una imagen, y sus respectivos keypoints
        y matches.
        """
        if len(good) > self.homography_min_match:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1 ,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
        
            h,w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

            if not isinstance(M, NoneType):
                M = M.astype(np.float32)
            dst = cv.perspectiveTransform(pts, M)
        
            img2 = cv.polylines(img2, [np.int32(dst)], True, color, 3, cv.LINE_AA)
        
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.homography_min_match))
            matchesMask = np.float32([])

        return matchesMask, img2