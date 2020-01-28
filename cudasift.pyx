# distutils: language = c++

from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "<utility>" namespace "std" nogil:
  T move[T](T) # don't worry that this doesn't quite match the c++ signature

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
    vector[float] GetFeatureDescriptors(SiftData &data);

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

# VectorWrapper holds a vector[float]
cdef class VectorWrapper:
    cdef Py_ssize_t ncols
    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef vector[float] vec

    # constructor and destructor are fairly unimportant now since
    # vec will be destroyed automatically.
    # def __cinit__(self):
    #    self.ncols = 128 

    cdef set_data(self, vector[float]& data):
        """
        Usage:
            cdef VectorWrapper w
            w.set_data(array) # "array" itself is invalid from here on
            numpy_array = np.asarray(w)
        """
        self.ncols = 128 
        # self.vec = move(data)
        # @ead suggests `self.vec.swap(data)` instead
        # to avoid having to wrap move
        self.vec.swap(data)

    # implement the buffer protocol for the class
    # which makes it generally useful to anything that expects an array
    def __getbuffer__(self, Py_buffer *buffer, int flags):
        # relevant documentation http://cython.readthedocs.io/en/latest/src/userguide/buffer.html#a-matrix-class
        cdef Py_ssize_t itemsize = sizeof(self.vec[0])

        # self.shape[0] = self.vec.size()
        self.shape[0] = self.vec.size() / self.ncols
        self.shape[1] = self.ncols

        #self.strides[0] = sizeof(int)
        # Stride 1 is the distance, in bytes, between two items in a row;
        # this is the distance between two adjacent items in the vector.
        # Stride 0 is the distance between the first elements of adjacent rows.
        self.strides[1] = <Py_ssize_t>(  <char *>&(self.vec[1])
                                       - <char *>&(self.vec[0]))
        self.strides[0] = self.ncols * self.strides[1]

        buffer.buf = <char *>&(self.vec[0])
        buffer.format = 'f'         # float
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.vec.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

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

    def get_feature_descriptors(self):
        """ TODO:
        - make VectorWrapper a member attribute of PyCudaSift, allocated only once
        - pass the vector[float] to GetFeatureDescriptors, one less call here in cython
        - Going to need fast access to the keypoints as well
        """
        cdef VectorWrapper v = VectorWrapper()
        cdef vector[float] dvec = GetFeatureDescriptors(self.c_sift_data)
        v.set_data(dvec)
        return v

def i_div_up(a, b):
    return iDivUp(a, b)

def i_div_down(a, b):
    return iDivDown(a, b)

def i_align_up(a, b):
    return iAlignUp(a, b)

def i_align_down(a, b):
    return iAlignDown(a, b)
