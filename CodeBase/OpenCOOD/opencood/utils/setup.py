from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os
from os.path import dirname, realpath, join, abspath

# 获取 setup.py 文件所在目录（绝对路径）
setup_dir = dirname(abspath(__file__))
pyx_file = join(setup_dir, 'box_overlaps.pyx')

# 切换到 setup.py 所在目录，确保输出文件在正确位置
original_dir = os.getcwd()
os.chdir(setup_dir)

try:
    ext_modules = [
        Extension(
            'box_overlaps',  # 模块名称
            ['box_overlaps.pyx'],  # 使用相对路径
            include_dirs=[numpy.get_include()]
        )
    ]
    
    setup(
        name='box overlaps',
        ext_modules=cythonize(ext_modules)
    )
finally:
    os.chdir(original_dir)