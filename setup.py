

from setuptools import setup

setup(
    name='pysparsdr',
    version='0.1.0',    
    description='SparSDR in python',
    url='https://github.com/ucsdwcsng/pySparSDR',
    author='Raghav Subbaraman',
    author_email='rsubbaraman@eng.ucsd.edu',
    license='Apache v2',
    packages=['pysparsdr'],
    install_requires=['scipy',
                      'numpy',
                      'matplotlib',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)