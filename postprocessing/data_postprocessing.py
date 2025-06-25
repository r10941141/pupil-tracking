import numpy as np
import cv2


def postprocessing_and_confidence(mask: np.ndarray, threshold_area: float = 4859 * 0.3) -> tuple:

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    if num_labels - 1 > 1:
        eroded = cv2.erode(dilated, kernel, iterations=7)
        dilated = cv2.dilate(eroded, kernel, iterations=7)

    nonzero_count = np.count_nonzero(dilated)
    confidence = 1.0 if nonzero_count > threshold_area else 0.0

    return dilated, confidence