
import sys
from setuptools import setup, setuptools
from mtcnn import __version__


__author__ = 'Iván de Paz Centeno'


def readme():
    with open('README.rst', encoding="UTF-8") as f:
        return f.read()


if sys.version_info < (3, 4, 1):
    sys.exit('Python < 3.4.1 is not supported!')


setup(name='mtcnn',
      version=__version__,
      description='Multi-task Cascaded Convolutional Neural Networks for Face Detection, based on TensorFlow',
      long_description=readme(),
      url='http://github.com/ipazc/mtcnn',
      author='Iván de Paz Centeno',
      author_email='ipazc@unileon.es',
      license='MIT',
      packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
      install_requires=[
      ],
      classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      keywords="mtcnn face detection tensorflow pip package",
      zip_safe=False)
