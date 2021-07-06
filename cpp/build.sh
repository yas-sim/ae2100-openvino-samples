mkdir -p build && pushd ./build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
popd
scp build/simple_cnn root@192.168.100.1:/home/root/tmp
