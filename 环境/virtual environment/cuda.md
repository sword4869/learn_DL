## cuda10.2

ubuntu22.04  gcc11.4.0

```bash
$ sudo ./cuda_10.2.89_440.33.01_linux.run 
 Failed to verify gcc version. See log at /var/log/cuda-installer.log for details.
```

那么遵从

```bash
$ sudo ./cuda_10.2.89_440.33.01_linux.run --override


 -   PATH includes /usr/local/cuda-10.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root

 cuda-10.2  cuda-11.8  
```

