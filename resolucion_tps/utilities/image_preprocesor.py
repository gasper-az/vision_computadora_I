import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import List, Tuple
from abc import ABC, abstractmethod

class PreprocessingStep(ABC):
    """
    Step genérico para nuestro pipeline de preprocesamiento.
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def preprocesar(self, image:cv.typing.MatLike) -> cv.typing.MatLike:
        pass

class ImagePreprocessor():
    """
    Aplica una serie de preprocessor steps a una imagen dada.
    """

    def __init__(self):
        pass

    def preprocesar(self, image:cv.typing.MatLike, steps:List[PreprocessingStep]) -> cv.typing.MatLike:
        """
        Aplica una serie de preprocesor steps a una imagen dada.

        Args:
            image (cv.typing.MatLike): imagen a preprocesar.
            steps (List[PreprocessingStep]): lista de preprocesor steps.
        Returns:
            (cv.typing.MatLike): imagen preprocesada.
        """
        image_to_process = image.copy()

        for step in steps:
            image_to_process = step.preprocesar(image=image_to_process)

        return image_to_process

class UnsharpMaskingStep(PreprocessingStep):
    """
    Esta clase permite aplicar Unsharp Masking a una imagen dada.
    """
    def __init__(self, sigma:float = 1.0, strength:float = 1.5):
        self.sigma = sigma
        self.strength = strength

    def preprocesar(self, image:cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Preprocesa una imagen mediante Unsharp Masking.

        Args:
            image (cv.typing.MatLike): imagen sobre la cual se aplica
                Unsharp Masking.
        Returns
            (cv.typing.MatLike): imagen con Unsharp Masking aplicado.
        """
        blurred = cv.GaussianBlur(image, (0,0), sigmaX=self.sigma)
        sharpened = cv.addWeighted(image, 1.0 + self.strength,
                                   blurred, -self.strength, 0)
        return sharpened

class ClaheStep(PreprocessingStep):
    """
    Esta clase permite aplicar CLAHE a una imagen dada.
    """
    def __init__(self, clipLimit: float = 2.0, tileGridSize:Tuple[int, int] = (8,8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def preprocesar(self, image:cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Preprocesa una imagen mediante CLAHE.

        Args:
            image (cv.typing.MatLike): imagen sobre la cual se aplica CLAHE.
        Returns
            (cv.typing.MatLike): imagen con CLAHE aplicado.
        """
        clahe = cv.createCLAHE(clipLimit=self.clipLimit,
                               tileGridSize=self.tileGridSize)
        result = clahe.apply(image)
        return result
    
class GaussianPyramidsStep(PreprocessingStep):
    """
    Esta clase permite aplicar Pirámides Gaussianas a una imagen dada.
    """
    def __init__(self, maxLevel:int = 1):
        self.maxLevel = maxLevel

    def preprocesar(self, image:cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Preprocesa una imagen aplicándole Pirámides Gaussianas.

        Args:
            image (cv.typing.MatLike): imagen sobre la cual se aplica
                Pirámides Gaussianas.
        Returns
            (cv.typing.MatLike): imagen correspondiente a la Pirámide Gaussiana
                de nivel N (self.maxLevel).
        """
        to_down = image

        for i in range(self.maxLevel):
            gaussianPyramid = cv.pyrDown(to_down)
            to_down = gaussianPyramid

        return gaussianPyramid

class CannyStep(PreprocessingStep):
    """
    Esta clase permite aplicar extracción de bordes por Canny a una imagen dada.
    """
    def __init__(self, lowerThreshold:int, upperThreshold:int):
        self.lowerThreshold = lowerThreshold
        self.upperThreshold = upperThreshold

    def preprocesar(self, image:cv.typing.MatLike) -> cv.typing.MatLike:
        """
        Preprocesa una imagen aplicándole Canny.

        Args:
            image (cv.typing.MatLike): imagen sobre la cual se aplica
                extracción de bordes de Canny.
        Returns
            (cv.typing.MatLike): imagen con bordes extraídos mediante Canny.
        """
        result = cv.Canny(image=image, threshold1=self.lowerThreshold,
                 threshold2=self.upperThreshold)
        return result