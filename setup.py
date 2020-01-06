from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

cudasift_extension = Extension(
    name="cudasift",
    sources=["cudasift.pyx"],
    language="c++",
    libraries=["cudasift"],
    library_dirs=["."],
    include_dirs=["."],
)
setup(
    name="cudasift",
    ext_modules=cythonize(cudasift_extension, build_dir="build"),
    script_args=["build"],
    options={"build": {"build_lib": "./build"}},
)
