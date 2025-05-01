from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# 定義要編譯的 Cython 擴充模組
extensions = [
    Extension(
        "yolo_cython_utils",
        ["yolo_cython_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]  # 使用高優化級別
    ),
]

# 設定編譯配置
setup(
    name="yolo_cython_utils",
    ext_modules=cythonize(extensions, language_level=3),
)
