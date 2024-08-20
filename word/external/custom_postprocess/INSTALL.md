# Installation

0. Create the environment from the environment.yml file.

    ~~~
    conda env create -f environment.yml
    ~~~    
1. Compile post processing module.

    ~~~
    python setup.py build_ext --inplace
    ~~~
2. Compile deformable convolutional (from [DCNv2](https://github.com/jinfagang/DCNv2_latest)).

    ~~~
    cd models/networks/DCNv2
    python3 setup.py build develop
    ~~~
3. Compile NMS.

    ~~~
    cd external
    make
    ~~~
