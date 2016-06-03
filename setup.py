import setuptools
import sys
import os

from numpy.distutils.core import Extension


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
    from numpy.distutils.core import setup

    setup(name="glmnet",
          version='0.1.0',
          description="Python wrapper for glmnet",
          long_description=read('README.md'),
          author="Bill Lattner",
          author_email="opensource@civisanalytics.com",
          ext_modules=[glmnet_lib],
          packages=['glmnet'],
          classifiers=[
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Programming Language :: Python :: 3',
              'Operating System :: OS Independent',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
              'Topic :: Scientific/Engineering'
          ])
