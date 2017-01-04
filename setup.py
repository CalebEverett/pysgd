from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pysgd',
    version='0.0.1a1',
    description='Stochastic gradient descent algorithms',
    long_description=long_description,
    url='https://github.com/CalebEverett/pysgd',
    download_url='https://github.com/CalebEverett/pysgd/archive/0.0.1a1.tar.gz',
    author='Caleb Everett',
    author_email='mail@calebeverett.io',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='machine-learning gradient-descent',
    packages=find_packages(),
    install_requires=['numpy'],
    extras_require={
        'test': ['pytest', 'py-cov', 'pyplot'],
    },
)
