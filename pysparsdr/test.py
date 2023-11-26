from tests.test_compressor import test_compression
"""
In this file, you can test the psSparSDR code with encoder added
run this file you will get the name of file that contains the binary data
then run N210_V1.py and input the file name

you can addjust the parameters in test_compression()
the file will be cleaned automaticlly when innitialize the pySparSDR class

"""
filename=test_compression()
print(filename)
