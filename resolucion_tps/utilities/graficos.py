import matplotlib.pyplot as plt

def graficar_imagen_e_histograma(imagen, histograma,
                                 cmap: any = "gray",
                                 vmin: float = 0,
                                 vmax: float = 255,
                                 title: str = "Imagen e histograma"):
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.imshow(imagen, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(title)

    ax2 = plt.subplot(212)
    ax2.plot(histograma)

def graficar_2_histogramas(histograma1, histograma2,
                         title1: str,
                         title2: str):
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(histograma1)
    ax1.set_title(title1)

    ax2 = plt.subplot(212)
    ax2.plot(histograma2)
    ax2.set_title(title2)