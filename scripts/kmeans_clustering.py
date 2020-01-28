#!/usr/bin/env python3
import argparse
import cudasift
import SiftGPU
import cv2
import numpy as np
import os
import random

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets import make_blobs

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial import Voronoi, voronoi_plot_2d

cs = cudasift.PyCudaSift(dev_num=0)
sg = SiftGPU.PySiftGPU()


def get_size(filename):
    st = os.stat(filename)
    return st.st_size


def compute_siftgpu_descriptors(filename, width, height):
    num_bytes = get_size(filename)
    with open(filename, "rb") as fp:
        data = fp.read()
    # SiftGPU
    # 0-3 with 3 being the most verbose
    sg.set_verbose(0)
    sg.parse_param(
        [
            "-p",  # Set WxH parameters for the pyramid (does no allocations)
            "1280x720",
            "-fo",  # First octave to detect DOG keypoints (default: 0, 1 will do: width >> 1)
            "-1",  # -1 here will: width << 1 and, height << 1
            "-lcpu",  # CPU/GPU mixed Feature List Generation (defaut : 6). Use GPU first, and use CPU when reduction size <= pow(2,num).
            "5",
            "-no",  # Max number of octaves (default: no limit)
            "5",
            "-loweo",
        ]
    )
    sg.create_context_gl()
    # Run
    sg.run_sift_buf(data, width, height, num_bytes)

    # Get data
    desc, kps = sg.copy_features()
    desc_np = np.asarray(desc)
    kps_np = np.asarray(kps)

    # L1 root normalize
    desc_np = root_sift(desc_np)

    cv2kps = []
    for i in range(len(kps_np)):
        cv2kps.append([kps_np[i][0], kps_np[i][1]])
        """
        cv2kps.append(
            cv2.KeyPoint(
                x=int(kps[i].get("x")),
                y=int(kps[i].get("y")),
                _size=kps[i].get("s"),
                _angle=kps[i].get("o"),
            )
        )
        """
    return desc_np, kps_np, np.array(cv2kps)


def root_sift(f):
    # L1 root normalize
    f = np.clip(f, 0, np.inf)
    f /= f.sum(axis=1, keepdims=True) + 1e-7
    f = np.sqrt(f)
    # scale by 512 and cast to uint8
    # f *= 512.0
    # f = f.astype(np.uint8)
    return f


def compute_cudasift_descriptors(data):

    # CudaSIFT
    height, width = data.shape
    data = data.astype(np.float32).ravel()
    # Allocate CUDA memory for the source image: once
    cs.allocate_cuda_image(
        width, height, cudasift.i_align_up(width, 128), False, None, None
    )
    cs.download_cuda_image(data)

    # Allocate storage for internal results
    cs.init_sift_data(max_pts=32768, host=True, dev=True)
    cs.allocate_sift_temp_memory(width, height, 5, False)

    # Run
    cs.extract_sift(
        # num_octaves=5, init_blur=1.0, thresh=2.0, lowest_scale=0.0, scale_up=False
        num_octaves=5,
        init_blur=1.0,
        thresh=2.0,
        lowest_scale=0.0,
        scale_up=False,
    )
    # Get descriptors: fast
    desc = cs.get_feature_descriptors()
    desc_np = np.asarray(desc)

    # L1 root normalize
    desc_np = root_sift(desc_np)

    # Add each single descriptor to the cluster
    # for i in range(desc_np.shape[0]):
    #    BOW_cudasift.add(desc_np[i])

    # Get (SiftData) keypoints: slow
    sift_data = cs.get_sift_data()
    desc = sift_data.get_sift_points()
    kps = []
    for i in range(sift_data.num_pts):
        kps.append([desc[i].get("xpos"), desc[i].get("ypos")])
        """
        kps.append(
            cv2.KeyPoint(
                x=int(desc[i].get("xpos")),
                y=int(desc[i].get("ypos")),
                _size=desc[i].get("scale"),
                _angle=desc[i].get("orientation"),
            )
        )
        """
    return desc_np, None, np.array(kps)


def get_kp_for_desc_siftgpu(centroid_desc, siftgpu_desc):
    # y is the row in the desc that is closest to the centroid
    y = pairwise_distances_argmin(centroid_desc.reshape(1, -1), siftgpu_desc)
    idx = y[0]
    # now find the corresponding keypoint
    kps, desc = sg.copy_feature_keypoints_descriptors()
    return (int(kps[idx].get("x")), int(kps[idx].get("y")))


def get_kp_for_desc_cudasift(centroid_desc, cudasift_desc):
    # y is the row in cudasift_desc that is closest to the centroid
    y = pairwise_distances_argmin(centroid_desc.reshape(1, -1), cudasift_desc)
    idx = y[0]
    # now find the corresponding keypoint
    desc = cs.get_sift_data().get_sift_points()
    return (int(desc[idx].get("xpos")), int(desc[idx].get("ypos")))


def add_plot(
    fig,
    kmeans_cluster,
    desc,
    keypoints,
    total_plots,
    ith_plot,
    n_clusters,
    cluster_desc,
    img,
):

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

    # Did we cluster the keypoints or the descriptors directly
    clustered_data = desc if cluster_desc else keypoints

    # closest contains the index of the point in desc that is closest to each centroid.
    # So closest[0] is the index of the closest point in desc to centroid 0, and
    # closest[1] is the index in desc closest to centroid 1 etc.
    closest, _ = pairwise_distances_argmin_min(
        kmeans_cluster.cluster_centers_, clustered_data
    )

    # The cluster in which each descriptor or keypoint lies.  That is:
    # labels is: array, shape [n_samples,] such that it contains
    # indices of the cluster each sample belongs to.
    labels = kmeans_cluster.predict(clustered_data)

    ax = fig.add_subplot(1, total_plots, ith_plot + 1)

    # underlay the input image
    ax.imshow(img, zorder=1, cmap=plt.cm.bone)

    for k in range(n_clusters):
        my_members = labels == k

        # Random color for this cluster
        col = np.random.rand(3)

        # print(f"k={k}, {len(keypoints[my_members])}, {keypoints[my_members].shape}")
        points = keypoints[my_members]
        hull = ConvexHull(points)
        # convex_hull_plot_2d(hull, ax=ax)
        ax.plot(
            points[:, 0],
            points[:, 1],
            "o",
            markerfacecolor=col,
            markeredgecolor=col,
            markersize=2,
        )
        # ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], color=col)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=col)

        """
        hull = Voronoi(keypoints[my_members])
        voronoi_plot_2d(
            hull,
            ax=ax,
            show_points=False,
            show_vertices=False,
            line_colors=random.choice(colors),
        )
        """
        # closest[k] is the *index* into the descriptors that is closest to the k-th
        # centroid.  So plot the 2D keypoint of this "quasi-centroid" descriptor
        centroid_kp = keypoints[closest[k]]

        """
        # Too many points?
        ax.plot(
            keypoints[my_members, 0],
            keypoints[my_members, 1],
            "w",
            markerfacecolor=col,
            # marker=".",
            markersize=2,
        )
        """
        ax.plot(
            centroid_kp[0],
            centroid_kp[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title(f"SiftGPU_{ith_plot}")
    # ax.set_title(f"CudaSift_{ith_plot}")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.axis("scaled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptors",
        default=False,
        action="store_true",
        help="Whether or not cluster feature descriptors",
    )
    args = parser.parse_args()

    filename = "../data/CY_279b46b9_1575825158217_1575825184058.jpg"
    data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    n_clusters = 20
    RUNS = 2

    fig = plt.figure(figsize=(8, RUNS))
    for i in range(RUNS):
        kmeans_cluster = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
        # Fit the keypoints
        # desc, _, cv2kps = compute_cudasift_descriptors(data)
        desc, kps, cv2kps = compute_siftgpu_descriptors(
            filename, data.shape[0], data.shape[1]
        )
        if args.descriptors:
            kmeans_cluster.fit(desc)
        else:
            kmeans_cluster.fit(cv2kps)

        # Add to plot
        add_plot(
            fig,
            kmeans_cluster,
            desc,
            cv2kps,
            RUNS,
            i,
            n_clusters,
            args.descriptors,
            data,
        )

    plt.show()


if __name__ == "__main__":
    main()
