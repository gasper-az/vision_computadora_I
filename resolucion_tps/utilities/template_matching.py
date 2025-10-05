import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class PreprocessingData:
    """
    Datos de configuración de procesamiento de imágenes.
    Permite definir lower y upper thresholds para Canny. Si ambos son cero, no se aplica Canny.
    Permite definier nivel deseado de Pirámides Gaussianas. Si es cero, no se calculan.
    """
    def __init__(self, cannyLowerThreshold: int = 0, cannyUpperThreshold: int = 0, gaussianPyramidsMaxLevel: int = 0):
        self.cannyLowerThreshold = cannyLowerThreshold
        self.cannyUpperThreshold = cannyUpperThreshold
        self.gaussianPyramidsMaxLevel = gaussianPyramidsMaxLevel

class TemplateMatching:
    """
    Realiza template matching sobre una imagen (target) y un template.
    """
    def __init__(self, image: cv.typing.MatLike, template: cv.typing.MatLike):

        self.originalImage = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
        
        # Default values para imágenes y templates procesados.
        self.processedImage = None
        self.processedTemplate = None

    def preprocesar(self, imagePreprocessingData: PreprocessingData, templatePreprocessingData: PreprocessingData):
        """
        Preprocesa con bordes de Canny y pirámides Gaussianas a la imagen target y al template.
        Orden de preprocesamiento:
            1. Aplicar Bordes de Canny sobre cada imagen.
            2. Aplicar Pirámides Gaussianas sobre Bordes de Canny.

        Args:
            imagePreprocessingData (PreprocessingData): Contiene la configuración a aplicar sobre la imagen target.
            templatePreprocessingData (PreprocessingData): Contiene la configuración a aplicar sobre el template.
        """

        # Si los lower y upper threshold son ceros, NO se aplica Bordes de Canny.
        if imagePreprocessingData.cannyLowerThreshold == 0 and imagePreprocessingData.cannyUpperThreshold == 0:
            imageCanny = self.image
        else:
            imageCanny = cv.Canny(self.image,
                                  threshold1=imagePreprocessingData.cannyLowerThreshold,
                                  threshold2=imagePreprocessingData.cannyUpperThreshold)
        
        # Si los lower y upper threshold son ceros, NO se aplica Bordes de Canny.            
        if templatePreprocessingData.cannyLowerThreshold == 0 and templatePreprocessingData.cannyUpperThreshold == 0:
            templateCanny = self.template
        else:
            templateCanny = cv.Canny(self.template,
                                  threshold1=templatePreprocessingData.cannyLowerThreshold,
                                  threshold2=templatePreprocessingData.cannyUpperThreshold)
        
        # No se aplica pirámides gaussianas si el valor es cero.
        if imagePreprocessingData.gaussianPyramidsMaxLevel == 0:
            imageGaussianPyramid = imageCanny
        else:    
            imageGaussianPyramid = self.__gaussianPyramids__(img=imageCanny,
                                                                  level=imagePreprocessingData.gaussianPyramidsMaxLevel)
        
        # No se aplica pirámides gaussianas si el valor es cero.
        if templatePreprocessingData.gaussianPyramidsMaxLevel == 0:
            templateGaussianPyramid = templateCanny
        else:    
            templateGaussianPyramid = self.__gaussianPyramids__(img=templateCanny,
                                                                  level=templatePreprocessingData.gaussianPyramidsMaxLevel)
        
        self.processedImage = imageGaussianPyramid
        self.processedTemplate = templateGaussianPyramid

    def templateMatching(self, metrica: int, threshold: float, template_title: str, target_title: str, final_title: str, graficar: bool = True) -> list:
        """
        Aplica template matching sobre la imagen y el template configurados.
        Source:
            1. https://pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/
            2. Código provisto en clase 4.

        Args:
            metrica (int): Métrica a aplicar. Debe ser alguna de las siguientes:
                https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
            threshold (float): Threshold para filtrar resultados.
            template_title (str): Título a utilizar al mostrar el template.
            target_title (str): Título a utilizar al mostrar la imagen target.
            final_title (str): Título a utilizar al mostrar los resultados.
            graficar (bool): Indica si se debe realizar un gráfico de resultados o no.
        
        Returns:
            list: lista de coordenadas que representan detecciones del template en la imagen de origen/target.

        Raises:
            TypeError: Si no se llamó anteriormente a preprocesar.
        """

        if self.processedImage is None or self.processedTemplate is None:
            raise TypeError("Imagen y o template NO configurados. Ejecutar preprocesar antes de llamar a este método.")
        
        img_salida = self.originalImage.copy()

        # Template Matching
        res = cv.matchTemplate(self.processedImage, self.processedTemplate, metrica)
        
        (tH, tW) = self.processedTemplate.shape[:2]

        # Filtrado de resultados
        if metrica in [cv.TM_SQDIFF_NORMED]:
            (yCoords, xCoords) = np.where( res <= threshold)
        else:
            (yCoords, xCoords) = np.where( res >= threshold)

        # Non Max Suppression
        rects = []
        
        for (x, y) in zip(xCoords, yCoords):
            rects.append((x, y, x + tW, y + tH))
        
        pick = self.__nonMaxSuppression__(np.array(rects))
        
        for (startX, startY, endX, endY) in pick:
            cv.rectangle(img_salida, (startX, startY), (endX, endY),
                (0, 0, 255), 2)

        if graficar:
            ### Gráficos
            plt.figure(figsize=(15,5))
            plt.subplot(131)
            plt.imshow(self.processedTemplate, cmap="gray")
            plt.title(template_title)

            plt.subplot(132)
            plt.imshow(self.processedImage, cmap="gray")
            plt.title(target_title)

            plt.subplot(133)
            plt.imshow(img_salida)
            plt.title(final_title)

        return pick
    
    def __gaussianPyramids__(self, img: cv.typing.MatLike, level: int = 2) -> cv.typing.MatLike:
        """
        Calcula las pirámides Gaussianas asociada a una imagen.

        Args:
            img (cv.typing.MatLike): Imagen base sobre la cual calcular las pirámides.
            level (int): Nivel de pirámides.
        Returns:
            cv.typing.MatLike: Pirámide gaussiana de nivel determinado.
        """

        to_down = img

        for i in range(level):
            gaussianPyramid = cv.pyrDown(to_down)
            to_down = gaussianPyramid

        return gaussianPyramid
    
    def __nonMaxSuppression__(self, boxes, overlapThresh=0.3):
        """
        Aplica Non Max Suppression.

        Sources:
            https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            https://github.com/PyImageSearch/imutils/blob/master/imutils/object_detection.py#L4
        """
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats.
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")