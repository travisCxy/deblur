from distutils.core import setup
from Cython.Build import cythonize

setup(name='Converter',
     ext_modules=cythonize('convert.py'))