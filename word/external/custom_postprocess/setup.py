import numpy as np  
from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Distutils import build_ext  


ext_modules = [Extension('post_process', ['post_process.pyx'], include_dirs=[np.get_include()]),]
for module in ext_modules:
    module.cython_directives = {'language_level': '3'}
setup(name='post_process', cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
