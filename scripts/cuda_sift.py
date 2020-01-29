#!/usr/bin/env python3
import cudasift
import cv2
import numpy as np

from profiling import TaggedTimer


def main():
    sift = cudasift.PyCudaSift(dev_num=0)
    timr = TaggedTimer()

    filename = "../data/CY_279b46b9_1575825158217_1575825184058.jpg"
    # filename = "/home/jfinken/projects/here/sp/jfinken/faiss_gpu/AIC_query2.jpg"
    data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # for writing out keypoints
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = data.shape
    print(f"Input image: {width}x{height}")
    # data = np.ascontiguousarray(data, dtype=np.float32).ravel()
    # data = np.array(data, dtype=np.float32).ravel()
    data = img.astype(np.float32).ravel()
    timr("np.ascontiguousarray")

    NUM_RUNS = 3

    # Allocate CUDA memory for the source image: once
    sift.allocate_cuda_image(
        width, height, cudasift.i_align_up(width, 128), False, None, None
    )
    timr("allocate_cuda_image")
    # Allocate storage for internal results
    sift.init_sift_data(max_pts=32768, host=True, dev=True)
    sift.allocate_sift_temp_memory(width, height, 5, False)
    timr("allocate_sift_temp_memory")

    for j in range(NUM_RUNS):

        # Convenient and temporally performant optimization:
        #   Reuse CUDA malloc-ed device memory
        #   Simply download this input image to the device
        sift.download_cuda_image(data)
        timr("download_cuda_image")

        # Run
        sift.extract_sift(
            # num_octaves=5, init_blur=1.0, thresh=2.0, lowest_scale=0.0, scale_up=False
            num_octaves=5,
            init_blur=1.0,
            thresh=2.0,
            lowest_scale=0.0,
            scale_up=False,
        )
        timr("extract_sift")
        print(timr)

        # Get descriptors and keypoints
        desc, kp = sift.get_features()
        desc_np = np.asarray(desc)
        kp_np = np.asarray(kp)
        timr(
            f"get_features done (num_pts={desc_np.shape[0]}, desc_np.shape={desc_np.shape}, kp_np.shape={kp_np.shape})"
        )
        print(timr)
        """
        # Debug: make cv2 keypoints
        kps = []
        for i in range(kp_np.shape[0]):
            # print(f"keypt @ {desc[i].get('xpos')}, {desc[i].get('ypos')}")
            kps.append(
                cv2.KeyPoint(
                    x=int(kp_np[i, 0]),
                    y=int(kp_np[i, 1]),
                    _size=kp_np[i, 2],
                    _angle=kp_np[i, 3],
                )
            )
        timr("for-loop over keypoints")
        print(timr)
        img = cv2.drawKeypoints(
            img,
            kps,
            outImage=np.array([]),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        timr("cv2.drawKeypoints")
        cv2.imwrite(f"woo.jpg", img)
        # timr("cv2.imwrite")
        """


if __name__ == "__main__":
    main()
