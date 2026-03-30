import cv2


def bf_lowe_match(desc1, desc2, ratio_threshold=0.75):
    """Match SIFT descriptors using brute-force L2 distance and Lowe's ratio test.

    For each descriptor in desc1, finds the two nearest neighbors in desc2.
    Keeps a match only when the closest distance is less than ratio_threshold
    times the second-closest distance.  This rejects ambiguous matches where
    two candidates in desc2 look nearly equally similar.

    Args:
        desc1 (np.ndarray | None): Descriptors from image 1, shape (N, 128), float32.
        desc2 (np.ndarray | None): Descriptors from image 2, shape (M, 128), float32.
        ratio_threshold (float): Lowe's ratio cutoff.  0.75 is the value from the
            original paper.  Lower values = stricter filtering = fewer but more
            reliable matches.

    Returns:
        list[cv2.DMatch]: Matches that passed the ratio test.  Empty list when
            descriptors are None or too short for knnMatch (need at least 2 each).
    """
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    return good
