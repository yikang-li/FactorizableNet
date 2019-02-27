all:
	python setup_cuda.py build develop
	python setup_cython.py build_ext --inplace 

clean:
	rm -rf nms/*.c nms/*.cpp  nms/*.so
	rm -rf utils/*.c utils/*.cpp utils/*.so
