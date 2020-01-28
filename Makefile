# Python 3, assert your env is active
cython:
	touch cudasift.pyx
	python setup.py build_ext --inplace

josh:
	@echo "Is site-packages in your LD_LIBRARY_PATH?  Also, that won't work on macOS anyhow..."
	cp build/cudasift.cpython-36m-x86_64-linux-gnu.so /home/jfinken/miniconda3/envs/py36/lib/python3.6/site-packages
	cp build/libcudasift.so /home/jfinken/miniconda3/envs/py36/lib/python3.6/site-packages

manual:
	cython --cplus -o build/cudasift.cpp cudasift.pyx
	gcc -pthread -B /home/jfinken/miniconda3/envs/py36/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I. -I/home/jfinken/miniconda3/envs/py36/include/python3.6m -c build/cudasift.cpp -o build/temp.linux-x86_64-3.6/build/cudasift.o
	g++ -pthread -shared -B /home/jfinken/miniconda3/envs/py36/compiler_compat -L/home/jfinken/miniconda3/envs/py36/lib -Wl,-rpath=/home/jfinken/miniconda3/envs/py36/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/build/cudasift.o -L. -lcudasift -o ./cudasift.cpython-36m-x86_64-linux-gnu.so
