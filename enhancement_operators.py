import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_cdf_hist(image_input):
    """
    Method to compute histogram and cumulative distribution function

    :param image_input: input image
    :return: cdf
    """
    hist, bins = np.histogram(image_input.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return cdf_normalized


def contrast_brightness(image, alpha, beta):
    """
    Linear transformation function to enhance brightness and contrast

    :param image: input image
    :param alpha: contrast factor
    :param beta: brightness factor
    :return: enhanced image
    """
    enhanced_image = np.array(alpha*image + beta)
    enhanced_image[enhanced_image > 255] = 255
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf


def gamma_enhancement(image, gamma):
    """
    Non-linear transformation function to enhance brightness and contrast
    :param image: input image
    :param gamma: contrast enhancement factor
    :return: enhanced image
    """
    normalized_image = image / np.max(image)
    enhanced_image = np.power(normalized_image, gamma)
    enhanced_image = enhanced_image * 255
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf


def log_enhancement(image, gain):
    """
    Non-linear transformation function to enhance brightness and contrast
    :param image: input image
    :param gain: contrast enhancement factor
    :return: enhanced image
    """
    normalized_image = image / np.max(image)
    enhanced_image = gain*np.log1p(normalized_image)
    enhanced_image = enhanced_image * 255
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf


def gauss_enhancement(image, gain):
    """
    Non-linear transformation function to enhance brightness and contrast
    :param image: input image
    :param gain: contrast enhancement factor
    :return: enhanced image
    """
    normalized_image = image / np.max(image)
    enhanced_image = 1 - np.exp(-normalized_image**2/gain)
    enhanced_image = enhanced_image*255
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf


def hist_enhancement(image):
    """
    Histogram equalization to enhance the input image
    :param image: input image
    :return: enhanced image
    """
    enhanced_image = cv2.equalizeHist(image)
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf


def clahe_enhancement(image, threshold, grid_size=(16, 16)):
    """
    Adaptive histogram equalization to enhance the input image
    :param image: input image
    :param threshold: clipping threshold
    :param grid_size: local neighbourhood
    :return: enhanced image
        """
    clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=grid_size)
    enhanced_image = clahe.apply(image)
    cdf = get_cdf_hist(enhanced_image)
    return enhanced_image, cdf


def image_enhancement_spatial():
    """
    Main method to change the pixels spatially
    :return: image grid
    """
    image_retina = plt.imread("slo_input.jpg")
    image_slo = plt.imread("retina_input.jpg")

    cdf_input = get_cdf_hist(image_slo)

    fig, axs = plt.subplots(5, 2, figsize=(8, 15))
    axs[0, 0].imshow(image_slo, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Input")
    axs[0, 1].hist(image_slo.flatten(), 256, [0, 256], color='r')
    axs[0, 1].plot(cdf_input)

    enhanced_cb, cdf_cb = contrast_brightness(image_slo, 1.6, 20)
    axs[1, 0].imshow(enhanced_cb, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title("Linear")
    axs[1, 1].hist(enhanced_cb.flatten(), 256, [0, 256], color='r')
    axs[1, 1].plot(cdf_cb)

    enhanced_gamma, cdf_gamma = gamma_enhancement(image_slo, 0.55)
    axs[2, 0].imshow(enhanced_gamma, cmap='gray', vmin=0, vmax=255)
    axs[2, 0].set_title("Non-linear (Gamma)")
    axs[2, 1].hist(enhanced_gamma.flatten(), 256, [0, 256], color='r')
    axs[2, 1].plot(cdf_gamma)

    enhanced_log, cdf_log = log_enhancement(image_slo, 1.65)
    axs[3, 0].imshow(enhanced_log, cmap='gray', vmin=0, vmax=255)
    axs[3, 0].set_title("Non-linear (Log)")
    axs[3, 1].hist(enhanced_log.flatten(), 256, [0, 256], color='r')
    axs[3, 1].plot(cdf_log)

    enhanced_gauss, cdf_gauss = gauss_enhancement(image_slo, 0.15)
    axs[4, 0].imshow(enhanced_gauss, cmap='gray', vmin=0, vmax=255)
    axs[4, 0].set_title("Non-linear (Inverse Gauss)")
    axs[4, 1].hist(enhanced_gauss.flatten(), 256, [0, 256], color='r')
    axs[4, 1].plot(cdf_gauss)

    for i in range(5):
        for j in range(2):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig("retina.png")
    plt.show()


def image_enhancement_spectral():
    """
    Main method to change the pixels based on distribution spectrum
    :return: image grid
    """

    image_slo = plt.imread("slo_input.jpg")
    image_retina = plt.imread("retina_input.jpg")

    cdf_input = get_cdf_hist(image_slo)

    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    axs[0, 0].imshow(image_slo, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Input")
    axs[1, 0].hist(image_slo.flatten(), 256, [0, 256], color='r')
    axs[1, 0].plot(cdf_input)

    enhanced_hist, cdf_hist = hist_enhancement(image_slo)
    axs[0, 1].imshow(enhanced_hist, cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Histogram Equalization")
    axs[1, 1].hist(enhanced_hist.flatten(), 256, [0, 256], color='r')
    axs[1, 1].plot(cdf_hist)

    enhanced_clahe, cdf_clahe = clahe_enhancement(image_slo, 10)
    axs[0, 2].imshow(enhanced_clahe, cmap='gray', vmin=0, vmax=255)
    axs[0, 2].set_title("CLAHE")
    axs[1, 2].hist(enhanced_clahe.flatten(), 256, [0, 256], color='r')
    axs[1, 2].plot(cdf_clahe)

    for i in range(2):
        for j in range(3):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.tight_layout()
    plt.show()


image_enhancement_spectral()
