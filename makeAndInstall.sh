nvcc -g --shared -o $PWD/lib/libleonard.so src/rbm.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcudart -lcublas --compiler-options '-fPIC' && \
./waf configure --prefix=$PWD --leonard-lib=$PWD/lib && \
./waf build && \
./waf install;
ctags -R src
