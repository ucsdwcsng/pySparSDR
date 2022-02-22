#/bin/python3 

import numpy as np
from scipy import signal as sig
from lib.pySparSDR import pySparSDRCompress

import matplotlib.pyplot as plt
import time

# Parameters for the compressor
nfft = 1024
numWindPerBlock = 10
signalIn = np.ones((nfft*numWindPerBlock,))
thresholdVec = 0.001*np.ones((1,nfft))

# Create a compressor object and initialize it
sparsdrObj = pySparSDRCompress(nfft, thresholdVec)

numIter = 10000
startTime = time.time()
for i in range(numIter):
    winIdx, binIdx, binVals = sparsdrObj.work(signalIn)
endTime = time.time()
print ("Time elapsed:", endTime - startTime)

# print(sparsdrObj.numWinProcessed);
megaSamplesPerSec = (signalIn.size * numIter)/(endTime - startTime)/np.power(10,6)
print ("Throughput: ", megaSamplesPerSec, " MSps")










