# pytorch-mkldnn-issue


### Build PyTorch using ExternalProject
```bash
export MAKEFLAGS=-j4

mkdir build && cd build
cmake3 -DCMAKE_BUILD_TYPE=Release -DBUILD_TORCH=ON ..
sudo make
```

### Build CartPole
```bash
# Remove old build directory
rm -rf build/

mkdir build && cd build
cmake3 -DCMAKE_BUILD_TYPE=Release -DBUILD_TORCH=OFF ..
sudo make
```
