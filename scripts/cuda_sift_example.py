#!/usr/bin/env python3
import cudasift
import cv2
import numpy as np


def main():
    sift = cudasift.PyCudaSift(dev_num=0)

    filename = "../data/img1.png"
    data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # See below: debug draw the keypoints
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = data.shape
    print(f"Input image: {width}x{height}")
    data = np.ascontiguousarray(data, dtype=np.float32).ravel()

    NUM_RUNS = 3

    # Allocate CUDA memory for the source image: once
    sift.allocate_cuda_image(
        width, height, cudasift.i_align_up(width, 128), False, None, data
    )

    for j in range(NUM_RUNS):

        # Convenient and temporally performant optimization:
        #   Reuse CUDA malloc-ed device memory
        #   Simply CUDA memcpy this input image to the device
        sift.download_cuda_image(data)

        # Allocate storage for internal results
        sift.init_sift_data(max_pts=32768, host=True, dev=True)
        sift.allocate_sift_temp_memory(width, height, 5, False)

        # Run
        sift.extract_sift(
            num_octaves=5, init_blur=1.0, thresh=2.0, lowest_scale=0.0, scale_up=False
        )

        # Get descriptors
        sift_data = sift.get_sift_data()
        desc = sift_data.get_sift_points()
        print(f"Num descriptors: {sift_data.num_pts}")

        # Debug: make and draw cv2 keypoints
        kps = []
        for i in range(sift_data.num_pts):
            kps.append(
                cv2.KeyPoint(
                    x=int(desc[i].get("xpos")),
                    y=int(desc[i].get("ypos")),
                    _size=desc[i].get("scale"),
                    _angle=desc[i].get("orientation"),
                )
            )
        img = cv2.drawKeypoints(
            img,
            kps,
            outImage=np.array([]),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(f"woot_{j}.jpg", img)


if __name__ == "__main__":
    main()
