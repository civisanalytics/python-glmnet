import sys
import os

# `Extension` from setuptools cannot compile Fortran code, so we have to use
# the one from numpy. To do so, we also need to use the `setup` function
# from numpy, not from setuptools.
# Confusingly, we need to explicitly import setuptools *before* importing
# from numpy, so that numpy can internally detect and use the `setup` function
# from setuptools.
# References: https://stackoverflow.com/a/51691203
#   and https://stackoverflow.com/a/55358607
import setuptools  # noqa: F401
try:
    from numpy.distutils.core import Extension, setup
except ImportError:
    sys.exit("install requires: 'numpy'."
             " use pip or easy_install."
             " \n  $ pip install numpy")


_VERSION = "2.2.0"

f_compile_args = ['-ffixed-form', '-fdefault-real-8']


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
        return _in.read()


def get_lib_dir(dylib):
    import subprocess
    from os.path import realpath, dirname

    p = subprocess.Popen("gfortran -print-file-name={}".format(dylib),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         shell=True)
    retcode = p.wait()
    if retcode != 0:
        raise Exception("Failed to find {}".format(dylib))

    libdir = dirname(realpath(p.communicate()[0].strip().decode('ascii')))

    return libdir


if sys.platform == 'darwin':
    GFORTRAN_LIB = get_lib_dir('libgfortran.3.dylib')
    QUADMATH_LIB = get_lib_dir('libquadmath.0.dylib')
    ARGS = ["-Wl,-rpath,{}:{}".format(GFORTRAN_LIB, QUADMATH_LIB)]
    f_compile_args += ARGS
    library_dirs = [GFORTRAN_LIB, QUADMATH_LIB]
else:
    library_dirs = None


glmnet_lib = Extension(name='_glmnet',
                       sources=['glmnet/_glmnet.pyf',
                                'glmnet/src/glmnet/glmnet5.f90'],
                       extra_f90_compile_args=f_compile_args,
                       library_dirs=library_dirs,
                       )

if __name__ == "__main__":
    setup(name="glmnet",
          version=_VERSION,
          description="Python wrapper for glmnet",
          long_description=read('README.rst'),
          long_description_content_type="text/x-rst",
          author="Civis Analytics Inc",
          author_email="opensource@civisanalytics.com",
          url="https://github.com/civisanalytics/python-glmnet",
          install_requires=[
              "numpy>=1.9.2",
              "scikit-learn>=0.18.0",
              "scipy>=0.14.1",
              "joblib>=0.14.1",
          ],
          python_requires=">=3.6.*",
          # We need pkg_resources, shipped with setuptools,
          # for version numbering.
          setup_requires=["setuptools"],
          ext_modules=[glmnet_lib],
          packages=['glmnet'],
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Environment :: Console',
              'Programming Language :: Python',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3 :: Only',
              'Operating System :: OS Independent',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
              'Topic :: Scientific/Engineering'
          ])
