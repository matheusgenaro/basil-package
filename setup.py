import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="basil.fit_statistics_cython",
        sources=["src/basil/fit_statistics_cython.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    package_dir={"": "src"},
    packages=["basil"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
