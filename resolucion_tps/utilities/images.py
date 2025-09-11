import cv2 as cv
import numpy as np

class ImageData:
    def __init__(self, path: str):
        self.path = path
        self.set_rgb_image()
        self.set_whitepatch()

    def set_rgb_image(self):
        self.rgbImage = leer_imagen_rgb(self.path)
    
    def set_whitepatch(self):
        self.whitePatchImage = whitepatch(self.rgbImage)


def leer_imagen_rgb(path: str) -> cv.typing.MatLike:
    img = cv.imread(path)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return imgRGB

def whitepatch(imgRGB: cv.typing.MatLike) -> cv.typing.MatLike:
    # Separo por canales
    red_channel, green_channel, blue_channel = cv.split(imgRGB)
    
    max_red = np.max(red_channel)
    max_green = np.max(green_channel)
    max_blue = np.max(blue_channel)

    # NOTE: no olvidarse de usar uint8
    whitePatchImg = cv.merge((
        np.uint8(255/max_red*red_channel),
        np.uint8(255/max_green*green_channel),
        np.uint8(255/max_blue*blue_channel)
    ))

    return whitePatchImg