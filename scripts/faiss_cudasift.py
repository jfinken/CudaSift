#!/usr/bin/env python3
import argparse
import cv2
import glob
import numpy as np
import os
import sys

import faiss
import cudasift
from profiling import TaggedTimer


""" Given CudaSift feature-descriptors, this example details FAISS approximate
nearest-neighbor indexing and querying (and optionally trains a new FAISS index).
"""

# QUERY_FILE = "AIC_query.jpg"
# INDEX_FILE = "./IVF3272-Flat-sift.faiss"
INDEX_FILE = "../data/CY_279b46b9_1575825158217-IndexIVFFlat-cudasift.faiss"


def root_sift(f):
    # L1 root normalize
    # f = np.clip(f, 0, np.inf)
    f /= f.sum(axis=1, keepdims=True) + 1e-7
    f = np.sqrt(f)
    # scale by 512 and cast to uint8
    f *= 512.0
    # TODO:
    #   the faiss query just converts it back to float32
    #   Line 453 of image_cloud_registration.py
    # f = f.astype(np.uint8)
    return f


def get_size(filename):
    st = os.stat(filename)
    return st.st_size


def train(idx_cpu, vec):
    # TODO: ask Landis, why back to float32
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)

    idx_cpu.train(vec)  # add vectors to the index
    idx_cpu.add(vec)  # add vectors to the index
    print(f"idx_cpu.ntotal={idx_cpu.ntotal}")
    return idx_cpu


def load_files(filenames):
    data = {}
    for filename in filenames:
        data[filename] = dict()
        _data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        _data = _data.astype(np.float32).ravel()
        data[filename]["data"] = _data
    return data


def train_faiss_index():
    w = 1280
    h = 720
    # Training data.  Path: FIXME
    datas = load_files(
        glob.glob("../../SiftGPU/data/CY_279b46b9_1575825158217_15758251*.jpg")
    )
    # Build a new index
    ndim = 128
    npts = 100000  # initial guess?
    faiss_type = f"IVF{int(4 * np.sqrt(npts))},Flat"
    idx_cpu = faiss.index_factory(ndim, faiss_type)

    # Construct the PyCudaSift object
    sift = cudasift.PyCudaSift(dev_num=0)

    # Allocate CUDA memory for the source image: once
    sift.allocate_cuda_image(w, h, cudasift.i_align_up(w, 128), False, None, None)

    # Allocate storage for internal results
    sift.init_sift_data(max_pts=32768, host=True, dev=True)
    sift.allocate_sift_temp_memory(w, h, 5, False)

    for data_file in datas.keys():

        data = datas[data_file]["data"]

        # Download this input image to the device
        sift.download_cuda_image(data)

        # Run
        sift.extract_sift(
            num_octaves=5, init_blur=1.0, thresh=2.0, lowest_scale=0.0, scale_up=False
        )
        # Get descriptors
        # sift_data = sift.get_sift_data()
        # desc = sift_data.get_sift_points()
        # print(f"Num descriptors: {sift_data.num_pts}")
        desc = sift.get_feature_descriptors()
        desc_np = np.asarray(desc)

        # L1 root normalize the desc
        desc_np = root_sift(desc_np)

        # Add the descriptors to the index
        train(idx_cpu, desc_np)

    faiss.write_index(idx_cpu, INDEX_FILE)


def main(train_faiss):
    if train_faiss:
        train_faiss_index()
        sys.exit(0)

    # Query data.  Path: FIXME
    datas = load_files(
        glob.glob("../../SiftGPU/data/CY_279b46b9_1575825158217_15758252*.jpg")
    )
    w = 1280
    h = 720

    timr = TaggedTimer()

    # Construct the PyCudaSift object
    sift = cudasift.PyCudaSift(dev_num=0)

    # Allocate CUDA memory for the source image: once
    sift.allocate_cuda_image(w, h, cudasift.i_align_up(w, 128), False, None, None)
    timr("allocate_cuda_image")

    # Allocate storage for internal results
    sift.init_sift_data(max_pts=32768, host=True, dev=True)
    sift.allocate_sift_temp_memory(w, h, 5, False)
    timr("allocate_sift_temp_memory")

    # Read the index
    idx_cpu = faiss.read_index(INDEX_FILE)
    timr("Reading the FAISS index")
    idx_gpu = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, idx_cpu)
    print(idx_gpu)
    timr("Download the FAISS index to the GPU")

    # SiftGPU
    for data_file in datas.keys():

        data = datas[data_file]["data"]

        # Download this input image to the device
        sift.download_cuda_image(data)
        timr("download_cuda_image")

        # Run
        sift.extract_sift(
            num_octaves=5, init_blur=1.0, thresh=2.0, lowest_scale=0.0, scale_up=False
        )
        timr("extract_sift!")
        # Get descriptors
        sift_data = sift.get_sift_data()
        # desc = sift_data.get_sift_points()
        # print(f"Num descriptors: {sift_data.num_pts}")
        desc = sift.get_feature_descriptors()
        # The numpy buffer interface is implemented within cython: no extra overhead
        desc_np = np.asarray(desc)

        # L1 root normalize the desc
        desc_np = root_sift(desc_np)

        timr(
            f"copy_features: num_features={sift_data.num_pts}, descriptors.shape={desc_np.shape}"
        )

        # Run the FAISS query
        k = 4
        lowe_ratio = 0.8
        sq_dist, idx = idx_gpu.search(desc_np, k)
        timr("Index query done")
        # Apply Lowe ratio test
        mask = sq_dist[:, 0] < (lowe_ratio ** 2) * sq_dist[:, 1]
        # mch: 2d array of (query index, db index) point correspondences
        mch = np.array(list(zip(mask.nonzero()[0], idx[mask, 0])), dtype=int)
        timr(f"Lowe ratio test done (len(mch)={len(mch)})")

        print(timr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="Whether or not to build a new FAISS index",
    )
    args = parser.parse_args()
    main(args.train)
