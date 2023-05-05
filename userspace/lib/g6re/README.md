# The G6 runtime environment


The G6 runtime environment implements C/C++ runtime to run guest applications to run on top of the G6 hypervisor.


## C runtime from musl


Please checkout the [g6re](http://10.3.2.105/haohui/musl) branch of the musl repository and build the musl C runtime with the following commands:

```
$ mkdir build
$ cddir build 
$ ../configure --disable-shared --prefix=<PREFIX>
$ make install
```

The library and the include files will be installed to the `PREFIX` directory.
 

## C++ runtime from libcxx


G6 uses the libcxx project from LLVM as the C++ runtime. Once you download the source code of LLVM 14.0.6. You can build the C++ runtime with the following command:

```
$ CXX=../clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang CC=../clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang cmake -G Ninja -S runtimes -B build \
-DCMAKE_INSTALL_PREFIX=<PREFIX> \
-DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
-DLIBCXX_ENABLE_SHARED=OFF -DLIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON -DLIBCXX_ENABLE_THREADS=OFF \
-DLIBCXX_ENABLE_RANDOM_DEVICE=OFF -DLIBCXX_ENABLE_UNICODE=OFF \
-DLIBCXX_HAS_MUSL_LIBC=ON \
-DLIBCXXABI_ENABLE_THREADS=OFF -DLIBCXX_ENABLE_EXCEPTIONS=OFF -DLIBCXXABI_ENABLE_EXCEPTIONS=OFF -DLIBCXXABI_ENABLE_THREADS=OFF
$ ninja -C build cxx cxxabi
$ ninja -C build install-cxx install-cxxabi
```

Ninja installs the library and the include files to the `PREFIX` directory

## Building the application

You can build the application with clang 14.0.6 on the x86-64 platform using the following command: 

```
clang -nostdinc -nostdlib -nostdinc++ -nostdlib++ -isystem <PREFIX>/include/c++/v1 -isystem <PREFIX>/include \
-fno-exceptions -fno-use-cxa-atexit \
<PREFIX>/Scrt1.o <PREFIX>/lib/crti.o main.cc
<PREFIX>/crtn.o <PREFIX>/lib/libc++.a <PREFIX>/lib/libc.a \
``` 

It should produce a static executable that can be run directly on Linux (assuming you are on the master branch of musl).

You need to specify `-mcmodel=kernel` if the application resides on the top 2GB address space. This is useful when testing the application directly on top of limine. 

