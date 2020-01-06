# distutils: language = c++

from libcpp cimport bool

cdef extern from "cudaImage.h":
    int iDivUp(int a, int b);
    int iDivDown(int a, int b);
    int iAlignUp(int a, int b);
    int iAlignDown(int a, int b);

cdef extern from "cudaImage.h":
    cdef cppclass CudaImage:
        void Allocate(int width, int height, int pitch, bool withHost, float *devMem, float *hostMem);
        double Download(float *hostMem);
        double Download();

cdef extern from "cudaSift.h":
    ctypedef struct SiftPoint:
        float xpos;
        float ypos;   
        float scale;
        float sharpness;
        float edgeness;
        float orientation;
        float score;
        float ambiguity;
        int match;
        float match_xpos;
        float match_ypos;
        float match_error;
        float subsampling;
        float empty[3];
        float data[128];
    ctypedef struct SiftData:
        int numPts
        SiftPoint *h_data;  # Host (CPU) data

    void InitCuda(int devNum);
    void InitSiftData(SiftData &data, int num, bool host, bool dev);
    float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp);
    void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory);
    void FreeSiftTempMemory(float *memoryTmp);

cdef class PySiftPoint:
    cdef SiftPoint *_ptr
    def __cinit__(self):
        self._ptr = NULL
    cdef from_ptr(self, SiftPoint* ptr):
        self._ptr = ptr
        return self

cdef class PySiftData:
    cdef SiftData *_ptr
    def __cinit__(self):
        self._ptr = NULL
    cdef from_ptr(self, SiftData* ptr):
        self._ptr = ptr
        return self
    def get_sift_points(self):
        """ This ultimately requires a patched version of cython.  See README
        for details.
        """
        cdef SiftPoint[:] p = <SiftPoint[:self.num_pts]>self._ptr.h_data
        return p
    @property
    def num_pts(self):
        return self._ptr.numPts


from cpython cimport array
cdef class PyCudaSift:
    cdef float *c_cuda_sift_temp_mem
    cdef CudaImage c_cuda_image
    cdef SiftData c_sift_data 
    cdef array.array dev_mem_arr
    cdef array.array host_mem_arr

    def __cinit__(self, dev_num=0):
        InitCuda(dev_num)

    def __dealloc__(self):
        # del self.c_cuda_image
        FreeSiftTempMemory(self.c_cuda_sift_temp_mem)

    def init_sift_data(self, max_pts, host, dev):
        InitSiftData(self.c_sift_data, max_pts, host, dev)

    def download_cuda_image(self, float[::1] host_mem):
        """From the host to the device.  NOTE: convenience function that
        obviates the need to call allocate_cuda_image with each new
        input image.  However, each input image must be the same 
        size (aspect ratio, etc).
        """
        return self.c_cuda_image.Download(&host_mem[0])
    
    def download_cuda_image(self):
        """From the host to the device.  NOTE: this function follows
        the API from the original CudaSift author.  This requires one
        to call allocate_cuda_image with each new input image (expensive).
        """
        return self.c_cuda_image.Download()

    def allocate_cuda_image(self, width, height, pitch, with_host, float[::1] dev_mem, float[::1] host_mem):
        """Somewhat ugly conditional logic but it avoids otherwise
        expensive and unnecessary memory allocations and/or copy
        operations here in cython, e.g. array.array('f', host_mem)

        Furthermore, memory is not accessed in CudaImage::Allocate anyhow.
        """
        if dev_mem is None and host_mem is None:
            self.c_cuda_image.Allocate(width, height, pitch, with_host, NULL, NULL)
        elif dev_mem is not None and host_mem is None:
            self.c_cuda_image.Allocate(width, height, pitch, with_host, &dev_mem[0], NULL)
        elif dev_mem is None and host_mem is not None:
            self.c_cuda_image.Allocate(width, height, pitch, with_host, NULL, &host_mem[0])
        else:
            self.c_cuda_image.Allocate(width, height, pitch, with_host, &dev_mem[0], &host_mem[0])


    def allocate_sift_temp_memory(self, width, height, num_octaves, scale_up):
        self.c_cuda_sift_temp_mem = AllocSiftTempMemory(width, height, num_octaves, scale_up)

    def extract_sift(self, num_octaves, init_blur, thresh, lowest_scale, scale_up):
        # Note the internal temp memory
        ExtractSift(self.c_sift_data, self.c_cuda_image, num_octaves, init_blur, thresh, lowest_scale, scale_up,  self.c_cuda_sift_temp_mem);

    def get_sift_data(self):
        return PySiftData().from_ptr(&self.c_sift_data)

def i_div_up(a, b):
    return iDivUp(a, b)

def i_div_down(a, b):
    return iDivDown(a, b)

def i_align_up(a, b):
    return iAlignUp(a, b)

def i_align_down(a, b):
    return iAlignDown(a, b)
