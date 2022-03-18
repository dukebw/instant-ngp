## Brendan's Notes

- See `build.sh` for my build script

- See `install.sh` for dependencies I needed on Fedora

- On gcc 11.2.1 and nvcc 11.6, I was getting `error: parameter packs not
  expanded with ‘...’`.
  I tried editing the C++ standard library source code to comment out the
  offending templates.
  See https://github.com/NVIDIA/nccl/issues/102#issuecomment-1021420403.
  But I still ran into compile errors.

- Ended up installing GCC 10.3.
  To do so I had to run:
  ```
  ./contrib/download_prerequisites
  ```
  in the gcc root.
  Then I had to install glibc-devel.i686 and libgcc.i686.

- I had to symlink my system libstdc++ into the GCC repository for some reason:
```
ln -s /usr/lib64/libstdc++.so.6.0.29 ./x86_64-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so
ln -s /usr/lib64/libstdc++.so.6.0.29 ./x86_64-pc-linux-gnu/libstdc++-v3/src/.libs/libstdc++.so.6
```
otherwise I got an error about `GLIB_CXX_3.4.29` not found.

- This excellent post explains a bit about building a custom libstdc++:
  https://stackoverflow.com/a/51946224/8581647

- This post explains how to build GCC from source:
  https://stackoverflow.com/a/10662297/8581647

- I had to symlink gcc and g++ to the CUDA installation in order for nvcc to
  find the correct headers, as described here:
  https://github.com/NVlabs/instant-ngp/issues/119#issuecomment-1034701258

- I had to install colmap from source to use ingp's colmap2nerf.py.
  The trickiest part of that was that I had to comment out lines 79-80 in
  `/usr/include/c++/11/type_traits`.
  See: https://github.com/colmap/colmap/issues/1418

- clang-format command line:
```
clang-format -i ./include/**/*.{cu,cuh,h,cpp}
clang-format -i ./src/**/*.{cu,cuh,h,cpp}
```
