# SIFT Feature Matching Under Real-World Conditions

This project tests SIFT feature matching on tourist photos from the Paris Buildings Dataset. I used 36 image pairs across four landmarks (Eiffel Tower, Louvre, Notre Dame, Sacré-Cœur) and measured how many descriptor matches survive geometric verification. The main question is which real-world conditions degrade performance most, and why some landmarks are far easier to match than others.

## Research Question

Which real-world condition most severely degrades SIFT feature matching performance, and why do some landmarks remain matchable while others fail?

## Dataset

The Paris Buildings Dataset contains Flickr tourist photos taken under uncontrolled conditions: mixed cameras, lighting, crowds, viewpoints.

I manually selected 36 image pairs across four landmarks and labeled each by visual inspection into one of four difficulty levels:

- **Easy** — similar angle and lighting
- **Medium** — different angle or different lighting
- **Hard** — day vs. night, heavy crowd occlusion, or extreme viewpoint shift
- **Stress** — heavily degraded pair intended to test the algorithm floor

These labels are subjective human judgments, not computed difficulty scores.

## How to Run

```bash
pip install -r requirements.txt
python3 -m src.pipeline
```

The pipeline runs a grid search over 48 parameter combinations, selects the best settings, re-runs all 36 pairs with those parameters, and saves results to `results/`.

## Method

The pipeline runs in this order:

1. Load image pairs from `pairs.json`
2. Convert to grayscale and apply Gaussian blur
3. Detect SIFT keypoints and descriptors via OpenCV `SIFT_create()`
4. Match descriptors using brute-force L2 with Lowe's ratio test
5. Run RANSAC homography to filter geometrically inconsistent matches
6. Record inlier count

The metric throughout is **RANSAC inlier count**: matches that survive both descriptor filtering and geometric verification. In the match visualizations, green lines are inliers and red lines are outliers. Some red lines may visually appear to connect correct points — this is expected. RANSAC fits one global homography across the whole image, so locally correct matches that don't fit the global transformation are still rejected as outliers, particularly where the building surface is non-planar or shooting angles differ slightly between images. Pairs with fewer than 4 matches after Lowe's ratio test automatically return 0 inliers, since RANSAC requires at least 4 point correspondences to estimate a homography.

## Parameters

Parameters were selected by grid search over 48 combinations (blur × ratio × reproj). The score function rewarded easy-hard separation per landmark, weighted 70% separation and 30% absolute easy strength.

| Parameter | Value | Reasoning |
|---|---|---|
| Gaussian blur kernel | (3, 3) | Light smoothing removes high-frequency noise without destroying the edge structure SIFT relies on |
| Gaussian blur sigma | 1.0 | Conservative; SIFT internally applies its own scale-space smoothing |
| Lowe's ratio threshold | 0.70 | Stricter than the Lowe 2004 default of 0.75; passes fewer matches but gives RANSAC cleaner, less ambiguous input |
| RANSAC reproj threshold | 8.0 px | Generous inlier radius handles wide-baseline pairs where the homography is only approximate |

The stricter ratio=0.70 outperformed the paper default: it produced fewer total matches but more geometrically consistent inliers, and unlocked the first successful Louvre match.

## Results

| Landmark | Easy avg inliers | Medium avg inliers | Hard avg inliers | Drop (easy → hard) |
|---|---|---|---|---|
| Notre Dame | 757 | 107 | 11 | 98% |
| Sacré-Cœur | 128 | 30 | 6.5 | 95% |
| Eiffel | 22 | 9 | 2 | 91% |
| Louvre | 4 | 11.5 | 2.3 | flat |

Notre Dame's easy average is pulled up by one near-identical pair (`notredame_easy_032_204`, 1502 inliers). Excluding it, the easy average drops to roughly 12; the degradation curve is still real but the raw number is not representative.

## Key Findings

Inlier count tracks difficulty well for Notre Dame and Sacré-Cœur. Notre Dame drops 98% from easy to hard (757 → 11). Sacré-Cœur drops 95% (128 → 6.5). The pattern breaks down for landmarks that don't generate distinctive descriptors to begin with.

The Louvre nearly fails at every difficulty level. With default parameters it produced zero good-green pairs. After tuning, one medium pair (`louvre_medium_050_022`, 19 inliers) worked; that was the only Louvre success across all 36 pairs. Two things compound: the facade is full of near-identical windows and columns, so Lowe's ratio test rejects most matches as ambiguous. And because the Louvre can be shot from many angles, background structures change completely between pairs, adding noise RANSAC can't resolve.

What determines performance is building surface and shooting context, not the difficulty label. Notre Dame and Sacré-Cœur have surfaces dense with corners, edges, and texture transitions (the local gradient structures SIFT detects). Gothic stonework, rose windows, and ornate domes produce descriptors that differ clearly from one patch to the next, so Lowe's ratio test passes matches with confidence. Their surroundings also stay consistent across photos: the Parvis plaza and Montmartre steps limit where tourists stand, so background structures match between pairs. Eiffel's lattice repeats the same diagonal-edge pattern across sections, which creates descriptor ambiguity. The Louvre has both problems: a repetitive facade and shooting positions that vary so much the background provides no stable reference.

Difficulty labels aren't always a reliable predictor. `eiffel_stress_091_030`, labeled as the most degraded category, produced 22 inliers, the same as the easy pair. The image happened to share more structure with the reference than the label implied. Labels reflect what a pair looks like visually, not how the algorithm will respond.

## Visualizations

`results/landmark_degradation.png` — line chart showing mean inlier count per landmark across easy → medium → hard. The core finding in one image: Notre Dame drops steeply (757 → 107 → 11), Louvre stays flat near zero at every level.

`results/difficulty_boxplot.png` — inlier distribution pooled by difficulty level across all 36 pairs, log scale with individual pair data points overlaid. Median decreases monotonically from easy to stress. Wide variance at easy and medium reflects the cross-landmark gap: Notre Dame's easy pairs and Louvre's easy pairs share the same label but differ by two orders of magnitude.

`results/inlier_summary.png` — 4-panel bar chart of RANSAC inlier counts for all 36 pairs, one panel per landmark, color-coded by difficulty. Log scale. The Louvre bars are nearly uniformly near-zero across all difficulty levels. The high outlier in Notre Dame is `notredame_easy_032_204` (1502 inliers).

`results/inlier_summary_merged.png` — same data as above but all four landmarks on a single axis with dashed separators between groups.

`results/match_viz/` — per-pair match images with green inlier lines and red outlier lines.

`results/final_good_greens/` — 14 pairs identified by visual review as clearly green-dominant after tuning.

## Project Structure

```
src/pipeline.py     main pipeline: loading, preprocessing, SIFT, matching, RANSAC, grid search
src/matchers.py     BFMatcher + Lowe's ratio test
src/geometric.py    RANSAC homography and inlier counting
src/visualize.py    match visualization and summary plots
pairs.json          36 image pair definitions across 4 landmarks and 4 difficulty levels
results/            generated outputs (plots, match images, CSV), included in repo
msml640_mid-term_project_proposal.pdf    original project proposal
```

## External-Source Policy

I used the following external sources/tools:

- OpenCV Documentation — https://docs.opencv.org/4.x/
- Paris Buildings Dataset — https://www.kaggle.com/datasets/skylord/oxbuildings — source of all 40 images used for evaluation
- LLM — used for coding assistance with `src/matchers.py`, `src/geometric.py`, and `src/visualize.py`. All pipeline logic, parameter choices, and analysis in `src/pipeline.py` are my own.
