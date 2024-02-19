import cv2 as cv


def canny_edge_detection(image, low_threshold=50, high_threshold=150, aperture_size=3, L2_gradient=False):
    edges = cv.Canny(image, low_threshold, high_threshold, apertureSize=aperture_size, L2gradient=L2_gradient)
    return edges
