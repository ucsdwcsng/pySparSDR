#/bin/python3 

import numpy as np
from scipy import signal as sig
from pysparsdr.pySparSDR import pySparSDRCompress

import matplotlib.pyplot as plt
import time

# Parameters for the compressor
nfft = 1024
numWindPerBlock = 100
thresholdVec = 0.001*np.ones((1,nfft))

# Create a compressor object and initialize it
sparsdrObj = pySparSDRCompress(nfft, thresholdVec)

numIter = 1000
signalIn = np.random.rand(nfft*numWindPerBlock,numIter)
startTime = time.time()
for i in range(numIter):
    winIdx, binIdx, binVals = sparsdrObj.work(signalIn[:,1])
endTime = time.time()
print ("Time elapsed:", endTime - startTime)

megaSamplesPerSec = (signalIn.size)/(endTime - startTime)/np.power(10,6)
print ("Throughput: ", megaSamplesPerSec, " MSps")










