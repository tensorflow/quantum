Instructions to create and intsall TFQuantum as a PIP package:

First open a docker with ubuntu 16.04 and the devtoolset7 toolchain.
```
./release/open_ubuntu_docker.sh
```
Then build pip packages for python 3.6 and 3.7 inside of the docker with:
```
./release/build_all_wheels.sh
exit
```

The resulting `.whl` files will be placed in a new folder outside of the docker called
`wheels`. From here they need to be repaired with `auditwheel` to ensure maximum
compatability across platforms. First open a manylinux2010 docker with:

```
./release/open_centos_docker.sh
```
Then clean the wheels:
```
./release/repair_wheels.sh
exit
```

Now the `wheels` folder should contain the built wheel files and the manylinux1
version of the wheels for python 3.5, 3.6 and 3.7.

A wheel can be installed with:
```
python3 -m pip install --user wheels/name_of_wheel.whl
```

Note, that if you are planning on running TFQuantum as a PIP package instead of
with Bazel, you cannot run from the TFQuantum/ directory, or python will attempt
to use the local files instead of the site-package/ files and fail to find
dependencies.
