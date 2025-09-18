import cv2 as cv
import numpy as np

class ImageQuality:
    def __init__(self):
        pass

    # Implementación del algoritmo definido en el Paper Image Sharpness MEasure for Blurred Images in Frequency.
    # Input: imagen de tamaño MxN
    # Output: medición de la calidad de la imagen.
    def imageQualityMeasure(self, image: cv.typing.MatLike, roiPercentage: float = 1.0) -> float:

        # Pasos del algoritmo:
        # Step 1: Compute F which is the Fourier Transform representation of image I   
        # Step 2: Find Fc which is obtained by shifting the origin of F to centre. 
        # Step 3: Calculate AF = abs (Fc) where AF is the absolute value of the centered Fourier transform of image I. 
        # Step 4: Calculate M = max (AF) where M is the maximum value of the frequency component in F. 
        # Step 5: Calculate TH = the total number of pixels in F whose pixel value > thres, where thres = M/1000. 
        # Step 6: Calculate Image Quality measure (FM) from equation TH/(MxN).

        if roiPercentage != 1.0:
            image = self.__getROIFromImage__(image=image, percentage=roiPercentage)

        # Paso 1: se computa F (transformada de fourier).
        F = np.fft.fft2(image)

        # Paso 2: se computa Fc (Se aplica shifting al centro de F).
        Fc = np.fft.fftshift(F)

        # Paso 3: se calcula abs de Fc.
        AF = np.abs(Fc)

        # Paso 4: se calcula max(AF).
        Max = np.max(AF)

        # Paso 5: se calcula TH, o la cantidad de pixeles en F cuyo valor es mayor a thres,
        # siendo thres = Max/1000.
        thres = Max/1000
        TH = F[F > thres].size

        ## TO-DO: la duda es utilizar F o AF???
        # TH = AF[AF > thres].size

        quality = TH/image.size

        return quality

    # Implementación de la métrica definida en el Paper Analysis of Focus Measure Operators.
    # Más info sobre esta métrica: https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/
    def tenengradFocusMeasure(self, image: cv.typing.MatLike, roiPercentage: float = 1.0) -> float:
        # Métrica = SUM_i_j(G_x(i, j)^2 + G_y(i, j)^2)
        # Donde
        #   G_x = magnitud del gradiente computados mediante la convolución de la imagen con operador de Sobel en X.
        #   G_y = magnitud del gradiente computados mediante la convolución de la imagen con operador de Sobel en Y.

        gray_image = image

        if len(image.shape) == 3:
            gray_image = image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Sobel function: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
        sobel_x = cv.Sobel(src=gray_image, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv.Sobel(src=gray_image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)

        tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)

        return np.mean(tenengrad)
    
    # Implementación de la métrica definida en el Paper Analysis of Focus Measure Operators.
    # Más info sobre esta métrica: https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/
    def brennerGradientFocusMeasure(self, image: cv.typing.MatLike, roiPercentage: float = 1.0) -> float:
        # Métrica = SUM_i_j(ABS(I(i,j) - I(i+2j))^2)
        #   I(i, j) es el pixel ubicado en la posición i, j
        #   I(i + 2j) es el pixel ubicado en dos posiciones de manera horizontal
        #   ABS = valor absoluto de la resta

        gray_image = image

        if len(image.shape) == 3:
            gray_image = image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Básicamente, es la imagen con los pixeles movidos dos unidades a la izquierda
        shifted = np.roll(gray_image, -2, axis=1)
        diferencia_imagenes = (gray_image - shifted) ** 2

        return np.sum(diferencia_imagenes)
    
    # Implementación de la métrica definida en el Paper Analysis of Focus Measure Operators.
    # Más info sobre esta métrica: https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/
    def varianceOfLaplacianFocusMeasure(self, image: cv.typing.MatLike, roiPercentage: float = 1.0) -> float:
        # Métrica = SUM_i_j(Delta(i, j) - Delta(I))^2
        #   Delta(I) es el valor medio del Laplaciano de la imagen

        gray_image = image

        if len(image.shape) == 3:
            gray_image = image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        laplaciano = cv.Laplacian(gray_image, ddepth=cv.CV_64F)

        return np.var(laplaciano)

    # Input: imagen de tamaño MxN; porcentaje para el ROI sobre el tamaño de imagen.
    # Output: ROI, en formato de imagen openCV.
    def __getROIFromImage__(self, image: cv.typing.MatLike, percentage: float) -> cv.typing.MatLike:
        (alto, ancho) = image.shape[:2]
        centro_y = alto//2
        centro_x = ancho//2

        alto_roi = int(alto*percentage)
        ancho_roi = int(ancho*percentage)

        start_y = max(0, centro_y - ancho_roi//2)
        start_x = max(0, centro_x - alto_roi//2)
        end_y = min(image.shape[1], centro_y + ancho_roi//2)
        end_x = min(image.shape[0], centro_x + alto_roi//2)

        roi = image[start_y:end_y, start_x:end_x]

        return roi