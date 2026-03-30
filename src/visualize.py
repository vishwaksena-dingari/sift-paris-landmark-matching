import os

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_DIFFICULTY_ORDER = ["easy", "medium", "hard", "stress"]
_DIFFICULTY_COLORS = {
    "easy":   "#4CAF50",
    "medium": "#FF9800",
    "hard":   "#F44336",
    "stress": "#9C27B0",
}


def draw_matches(img1, kp1, img2, kp2, good_matches, inlier_mask=None,
                 title="", save_path=None, max_draw=60):
    """Render matched keypoints side-by-side with inliers in green, outliers in red.

    When inlier_mask is provided the mask is used to colour-code matches.
    When omitted, all matches are drawn in green.

    Args:
        img1, img2 (np.ndarray): BGR images from cv2.imread.
        kp1, kp2 (list[cv2.KeyPoint]): Keypoints from SIFT.
        good_matches (list[cv2.DMatch]): Matches after Lowe's ratio test.
        inlier_mask (np.ndarray | None): Per-match uint8 mask from count_inliers.
        title (str): Plot title shown above the image.
        save_path (str | None): File path to save figure; None = show interactively.
        max_draw (int): Cap on lines drawn — too many lines are illegible.

    Returns:
        np.ndarray: The annotated BGR comparison image.
    """
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    if inlier_mask is not None and len(inlier_mask) == len(good_matches):
        inlier_list  = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
        outlier_list = [good_matches[i] for i in range(len(good_matches)) if not inlier_mask[i]]

        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2,
            inlier_list[:max_draw], None,
            matchColor=(0, 200, 0),
            singlePointColor=(180, 180, 180),
            flags=flags,
        )
        if outlier_list:
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2,
                outlier_list[:max_draw], match_img,
                matchColor=(0, 0, 220),
                singlePointColor=None,
                flags=flags | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
            )
    else:
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2,
            good_matches[:max_draw], None,
            matchColor=(0, 200, 0),
            singlePointColor=(180, 180, 180),
            flags=flags,
        )

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=12, pad=8)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return match_img


def plot_inlier_summary(results, save_path=None):
    """Bar chart: inlier counts per pair, grouped by landmark, colour-coded by difficulty.

    Args:
        results (list[dict]): Each dict must have keys:
            "pair_id" (str), "landmark" (str), "difficulty" (str), "inlier_count" (int).
        save_path (str | None): Save to file when given; otherwise show interactively.
    """
    if not results:
        return

    landmarks = sorted({r["landmark"] for r in results})
    ncols = len(landmarks)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, lm in zip(axes, landmarks):
        lm_results = sorted(
            [r for r in results if r["landmark"] == lm],
            key=lambda r: (_DIFFICULTY_ORDER.index(r["difficulty"])
                           if r["difficulty"] in _DIFFICULTY_ORDER else 99,
                           r["pair_id"]),
        )

        labels  = [r["pair_id"].split("_", 2)[-1] + f"\n({r['difficulty']})"
                   for r in lm_results]
        heights = [r["inlier_count"] for r in lm_results]
        colors  = [_DIFFICULTY_COLORS.get(r["difficulty"], "#607D8B") for r in lm_results]

        # Shorten labels: just the image index numbers and difficulty
        short_labels = []
        for r in lm_results:
            parts = r["pair_id"].split("_")
            # e.g. "eiffel_easy_091_005" → "091/005\n(easy)"
            nums = "_".join(parts[-2:]) if len(parts) >= 2 else r["pair_id"]
            short_labels.append(f"{nums}\n({r['difficulty'][:3]})")

        # Clip zeros to 0.5 so log bars render
        plot_heights = [max(h, 0.5) for h in heights]

        bars = ax.bar(range(len(short_labels)), plot_heights, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(lm.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Inlier count (log)" if ax is axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x)}" if x >= 1 else "0")
        )

        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(h, 0.5) * 1.15,
                    str(h), ha="center", va="bottom", fontsize=7)

    legend_patches = [
        mpatches.Patch(color=_DIFFICULTY_COLORS[d], label=d.capitalize())
        for d in _DIFFICULTY_ORDER
    ]
    fig.legend(handles=legend_patches, loc="upper right", framealpha=0.9)
    fig.suptitle("SIFT Inlier Counts by Difficulty Level", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_inlier_summary_merged(results, save_path=None):
    """Single merged bar chart: all pairs on one axis, grouped by landmark.

    Bars are color-coded by difficulty. A small gap and dashed line separate
    landmark groups. Y-axis is log scale so the Notre Dame outlier does not
    flatten everything else. Use this for slide/presentation contexts where
    a single image is preferred over the 4-subplot version.

    Args:
        results (list[dict]): Same format as plot_inlier_summary.
        save_path (str | None): Save to file when given; otherwise show interactively.
    """
    if not results:
        return

    landmarks = sorted({r["landmark"] for r in results})
    GAP = 1.5  # extra x-space between landmark groups

    bar_positions = []
    bar_heights   = []
    bar_colors    = []
    tick_labels   = []
    group_centers = {}
    separator_xs  = []

    x = 0.0
    for lm in landmarks:
        lm_results = sorted(
            [r for r in results if r["landmark"] == lm],
            key=lambda r: (
                _DIFFICULTY_ORDER.index(r["difficulty"])
                if r["difficulty"] in _DIFFICULTY_ORDER else 99,
                r["pair_id"],
            ),
        )
        start_x = x
        for r in lm_results:
            bar_positions.append(x)
            bar_heights.append(r["inlier_count"])
            bar_colors.append(_DIFFICULTY_COLORS.get(r["difficulty"], "#607D8B"))
            parts = r["pair_id"].split("_")
            tick_labels.append("_".join(parts[-2:]) if len(parts) >= 2 else r["pair_id"])
            x += 1.0
        group_centers[lm] = (start_x + x - 1) / 2.0
        separator_xs.append(x - 1 + GAP / 2)
        x += GAP

    fig, ax = plt.subplots(figsize=(14, 5))

    plot_heights = [max(h, 0.5) for h in bar_heights]
    ax.bar(bar_positions, plot_heights, color=bar_colors,
           edgecolor="white", linewidth=0.5, width=0.8)

    for pos, h, ph in zip(bar_positions, bar_heights, plot_heights):
        ax.text(pos, ph * 1.25, str(h),
                ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6.5)

    for lm, cx in group_centers.items():
        ax.text(cx, -0.22, lm.replace("_", " ").title(),
                ha="center", va="top", fontsize=10, fontweight="bold",
                transform=ax.get_xaxis_transform())

    for sx in separator_xs[:-1]:
        ax.axvline(sx, color="#BDBDBD", linewidth=1, linestyle="--")

    ax.set_yscale("log")
    ax.set_ylabel("RANSAC Inlier Count (log scale)", fontsize=11)
    ax.set_title("SIFT Inlier Counts: All 36 Pairs Grouped by Landmark", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}" if x >= 1 else "0")
    )

    legend_patches = [
        mpatches.Patch(color=_DIFFICULTY_COLORS[d], label=d.capitalize())
        for d in _DIFFICULTY_ORDER
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_difficulty_boxplot(results, save_path=None):
    """Boxplot comparing RANSAC inlier distributions across difficulty levels.

    Uses a log y-axis so the Notre Dame outlier (1502 inliers) does not flatten
    all other boxes. Individual pair data points are overlaid with jitter.

    Args:
        results (list[dict]): Same format as plot_inlier_summary.
        save_path (str | None): Save to file when given; otherwise show interactively.
    """
    grouped = {
        d: [r["inlier_count"] for r in results if r["difficulty"] == d]
        for d in _DIFFICULTY_ORDER
    }
    present = [d for d in _DIFFICULTY_ORDER if grouped[d]]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Clip zeros to 0.5 so log scale doesn't break
    data_for_plot = [
        [max(v, 0.5) for v in grouped[d]] for d in present
    ]

    bp = ax.boxplot(
        data_for_plot,
        labels=[d.capitalize() for d in present],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="", linestyle="none"),
    )
    for patch, d in zip(bp["boxes"], present):
        patch.set_facecolor(_DIFFICULTY_COLORS[d])
        patch.set_alpha(0.55)

    # Overlay individual data points with jitter
    rng = np.random.default_rng(42)
    for i, d in enumerate(present):
        vals = [max(v, 0.5) for v in grouped[d]]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.array([i + 1] * len(vals)) + jitter,
            vals,
            color=_DIFFICULTY_COLORS[d],
            edgecolors="white",
            linewidths=0.5,
            s=40,
            zorder=3,
            alpha=0.85,
        )

    # Annotate median value on each box
    for i, d in enumerate(present):
        vals = grouped[d]
        if vals:
            median_val = sorted(vals)[len(vals) // 2]
            ax.text(
                i + 1, max(median_val, 0.5) * 1.25,
                f"med={median_val}",
                ha="center", va="bottom", fontsize=8, color="black",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Difficulty Level", fontsize=11)
    ax.set_ylabel("RANSAC Inlier Count (log scale)", fontsize=11)
    ax.set_title("Inlier Distribution by Difficulty: All 36 Pairs", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x)}" if x >= 1 else "0")
    )
    fig.text(
        0.99, 0.01,
        "* Zeros plotted at 0.5 for log scale. "
        "notredame_easy_032_204 (1502 inliers) is the top outlier.",
        ha="right", va="bottom", fontsize=7, color="gray",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_landmark_degradation(results, save_path=None):
    """Line chart showing average inlier count per landmark across difficulty levels.

    X-axis: easy → medium → hard (stress excluded — too few pairs per landmark).
    Y-axis: mean RANSAC inlier count on a log scale.
    One line per landmark. Shows which landmarks degrade steeply vs stay flat.

    Args:
        results (list[dict]): Same format as plot_inlier_summary.
        save_path (str | None): Save to file when given; otherwise show interactively.
    """
    difficulties = ["easy", "medium", "hard"]
    landmarks = sorted({r["landmark"] for r in results})

    _LANDMARK_COLORS = {
        "eiffel":     "#1976D2",
        "louvre":     "#E65100",
        "notredame":  "#2E7D32",
        "sacrecoeur": "#6A1B9A",
    }
    _LANDMARK_MARKERS = {
        "eiffel":     "o",
        "louvre":     "s",
        "notredame":  "^",
        "sacrecoeur": "D",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for lm in landmarks:
        means = []
        xs = []
        for diff in difficulties:
            vals = [r["inlier_count"] for r in results
                    if r["landmark"] == lm and r["difficulty"] == diff]
            if vals:
                means.append(max(sum(vals) / len(vals), 0.5))
                xs.append(diff.capitalize())

        color  = _LANDMARK_COLORS.get(lm, "#607D8B")
        marker = _LANDMARK_MARKERS.get(lm, "o")
        label  = lm.replace("_", " ").title()
        ax.plot(xs, means, marker=marker, color=color, linewidth=2.2,
                markersize=9, label=label)

        # Annotate the easy value
        if means:
            ax.annotate(
                f"{means[0]:.0f}",
                (xs[0], means[0]),
                textcoords="offset points", xytext=(-18, 4),
                fontsize=8, color=color,
            )

    ax.set_yscale("log")
    ax.set_ylabel("Mean RANSAC Inlier Count (log scale)", fontsize=11)
    ax.set_xlabel("Difficulty Level", fontsize=11)
    ax.set_title("Inlier Drop-off by Landmark: Easy → Medium → Hard", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(round(x))}" if x >= 1 else "<1")
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.text(
        0.99, 0.01,
        "notredame easy mean includes 1502-inlier outlier pair.",
        ha="right", va="bottom", fontsize=7, color="gray",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
