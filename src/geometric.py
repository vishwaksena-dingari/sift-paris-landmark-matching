import cv2
import numpy as np


def count_inliers(kp1, kp2, good_matches, reproj_threshold=5.0):
    """Filter descriptor matches with RANSAC and return the inlier count.

    Extracts pixel coordinates from matched keypoints, calls cv2.findHomography
    with RANSAC, and counts how many matches survive (inliers).  A homography
    requires at least 4 point correspondences; fewer matches return zero inliers.

    Args:
        kp1 (list[cv2.KeyPoint]): Keypoints detected in image 1.
        kp2 (list[cv2.KeyPoint]): Keypoints detected in image 2.
        good_matches (list[cv2.DMatch]): Filtered matches from bf_lowe_match.
        reproj_threshold (float): Maximum allowed reprojection error (pixels) for
            a match to be classified as an inlier.  5.0 px is a standard default.

    Returns:
        dict:
            "inlier_count"  (int)         — number of geometrically consistent matches.
            "total_matches" (int)         — total matches passed to RANSAC.
            "inlier_mask"   (np.ndarray | None) — per-match boolean mask from RANSAC.
            "homography"    (np.ndarray | None) — 3×3 H matrix, or None if RANSAC failed.
    """
    result = {
        "inlier_count": 0,
        "total_matches": len(good_matches),
        "inlier_mask": None,
        "homography": None,
    }

    if len(good_matches) < 4:
        return result

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)

    if mask is not None:
        result["inlier_count"] = int(np.sum(mask))
        result["inlier_mask"] = mask
        result["homography"] = H

    return result
