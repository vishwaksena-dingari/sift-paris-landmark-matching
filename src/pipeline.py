import os
import json
import csv

import cv2

from src.matchers import bf_lowe_match
from src.geometric import count_inliers
from src.visualize import (
    draw_matches,
    plot_inlier_summary,
    plot_inlier_summary_merged,
    plot_difficulty_boxplot,
    plot_landmark_degradation,
)

# -------------------------
# Tunable parameters
# -------------------------
# Default values — overridden by tune_parameters() when tuning mode runs
BLUR_KERNEL = (3, 3)
BLUR_SIGMA = 1.0
RATIO_THRESHOLD = 0.75
REPROJ_THRESHOLD = 5.0

PAIRS_JSON = "pairs.json"
DATA_DIR = "data"
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
MATCHES_DIR = os.path.join(RESULTS_DIR, "match_viz")


def preprocess_image(img_bgr, blur_kernel, blur_sigma):
    """Convert BGR image to grayscale and optionally apply Gaussian blur."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_kernel is not None:
        gray = cv2.GaussianBlur(gray, blur_kernel, blur_sigma)
    return gray


def load_pairs(json_path):
    """Load pair metadata from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def save_results_csv(results, csv_path):
    """Save list of dict results to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "pair_id",
        "landmark",
        "difficulty",
        "match_count",
        "inlier_count",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_pipeline_once(
    pairs,
    blur_kernel,
    blur_sigma,
    ratio_threshold,
    reproj_threshold,
    save_visualizations=False,
):
    """Run the SIFT pipeline once for one parameter setting."""
    sift = cv2.SIFT_create()
    results = []

    for pair in pairs:
        pair_id = pair["id"]
        landmark = pair["landmark"]
        difficulty = pair["difficulty"]

        img1_path = os.path.join(DATA_DIR, landmark, pair["reference"])
        img2_path = os.path.join(DATA_DIR, landmark, pair["test"])

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"[WARN] Could not read images for pair {pair_id}")
            results.append({
                "pair_id": pair_id,
                "landmark": landmark,
                "difficulty": difficulty,
                "match_count": 0,
                "inlier_count": 0,
            })
            continue

        proc1 = preprocess_image(img1, blur_kernel, blur_sigma)
        proc2 = preprocess_image(img2, blur_kernel, blur_sigma)

        kp1, desc1 = sift.detectAndCompute(proc1, None)
        kp2, desc2 = sift.detectAndCompute(proc2, None)

        good_matches = bf_lowe_match(
            desc1,
            desc2,
            ratio_threshold=ratio_threshold,
        )

        geom_result = count_inliers(
            kp1,
            kp2,
            good_matches,
            reproj_threshold=reproj_threshold,
        )

        inlier_count = geom_result["inlier_count"]

        results.append({
            "pair_id": pair_id,
            "landmark": landmark,
            "difficulty": difficulty,
            "match_count": len(good_matches),
            "inlier_count": inlier_count,
        })

        if save_visualizations:
            save_match_path = os.path.join(MATCHES_DIR, f"{pair_id}.png")
            draw_matches(
                img1,
                kp1,
                img2,
                kp2,
                good_matches,
                inlier_mask=geom_result["inlier_mask"].ravel()
                if geom_result["inlier_mask"] is not None else None,
                title=(
                    f"{pair_id} | {landmark} | {difficulty} | "
                    f"inliers={inlier_count}"
                ),
                save_path=save_match_path,
            )

        print(
            f"{pair_id}: landmark={landmark}, difficulty={difficulty}, "
            f"matches={len(good_matches)}, inliers={inlier_count}"
        )

    return results


def score_results(results):
    """Average per-landmark score combining easy-hard separation and easy-match strength."""
    landmarks = sorted({r["landmark"] for r in results})
    scores = []

    for lm in landmarks:
        lm_results = [r for r in results if r["landmark"] == lm]

        easy = [r["inlier_count"] for r in lm_results if r["difficulty"] == "easy"]
        hard = [r["inlier_count"] for r in lm_results if r["difficulty"] == "hard"]

        if not easy or not hard:
            continue

        easy_mean = sum(easy) / len(easy)
        hard_mean = sum(hard) / len(hard)

        if easy_mean <= 0:
            scores.append(0.0)
            continue

        separation = max(0.0, (easy_mean - hard_mean) / easy_mean)
        strength = easy_mean / (easy_mean + 20.0)
        score = 0.7 * separation + 0.3 * strength
        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def tune_parameters(pairs):
    """Grid search over parameter settings using average per-landmark easy-hard separation."""
    blur_options = [None, (3, 3), (5, 5)]
    ratio_options = [0.70, 0.75, 0.80, 0.85]
    reproj_options = [3.0, 4.5, 5.0, 8.0]

    best_params = None
    best_score = float("-inf")
    all_trials = []

    for blur_kernel in blur_options:
        for ratio_threshold in ratio_options:
            for reproj_threshold in reproj_options:
                print(
                    "\n[TUNE] Testing:",
                    f"blur={blur_kernel}, "
                    f"ratio={ratio_threshold}, "
                    f"reproj={reproj_threshold}"
                )

                results = run_pipeline_once(
                    pairs=pairs,
                    blur_kernel=blur_kernel,
                    blur_sigma=1.0,
                    ratio_threshold=ratio_threshold,
                    reproj_threshold=reproj_threshold,
                    save_visualizations=False,
                )

                score = score_results(results)

                all_trials.append({
                    "blur_kernel": list(blur_kernel) if blur_kernel is not None else None,
                    "blur_sigma": 1.0,
                    "ratio_threshold": ratio_threshold,
                    "reproj_threshold": reproj_threshold,
                    "score": score,
                })

                print(f"[TUNE] Score = {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_params = {
                        "blur_kernel": blur_kernel,
                        "blur_sigma": 1.0,
                        "ratio_threshold": ratio_threshold,
                        "reproj_threshold": reproj_threshold,
                    }

    print("\n[TUNE] Best parameters found:")
    print(best_params)
    print(f"[TUNE] Best score = {best_score:.3f}")

    return best_params, all_trials

def save_tuning_csv(trials, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "blur_kernel",
        "blur_sigma",
        "ratio_threshold",
        "reproj_threshold",
        "score",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MATCHES_DIR, exist_ok=True)

    pairs = load_pairs(PAIRS_JSON)["pairs"]

    best_params, all_trials = tune_parameters(pairs)

    with open(os.path.join(RESULTS_DIR, "tuning_trials.json"), "w") as f:
        json.dump(all_trials, f, indent=2)
    
    save_tuning_csv(all_trials, os.path.join(RESULTS_DIR, "tuning_trials.csv"),)
        
    with open(os.path.join(RESULTS_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    results = run_pipeline_once(
        pairs=pairs,
        blur_kernel=best_params["blur_kernel"],
        blur_sigma=best_params["blur_sigma"],
        ratio_threshold=best_params["ratio_threshold"],
        reproj_threshold=best_params["reproj_threshold"],
        save_visualizations=True,
    )

    save_results_csv(results, RESULTS_CSV)

    plot_inlier_summary(
        results,
        save_path=os.path.join(RESULTS_DIR, "inlier_summary.png"),
    )
    plot_inlier_summary_merged(
        results,
        save_path=os.path.join(RESULTS_DIR, "inlier_summary_merged.png"),
    )
    plot_difficulty_boxplot(
        results,
        save_path=os.path.join(RESULTS_DIR, "difficulty_boxplot.png"),
    )
    plot_landmark_degradation(
        results,
        save_path=os.path.join(RESULTS_DIR, "landmark_degradation.png"),
    )

    print(f"\nSaved CSV to: {RESULTS_CSV}")
    print(f"Saved plots to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()