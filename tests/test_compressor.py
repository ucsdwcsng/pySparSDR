import numpy as np
import time
from pysparsdr.pySparSDR import pySparSDRCompress


def test_compression():
    print()
    print(f"Running compressor on random data")

    # Parameters for the compressor
    nfft = 1024
    numWindPerBlock = 100
    thresholdVec = 0.001 * np.ones((1, nfft))

    # Create a compressor object and initialize it
    sparsdrObj = pySparSDRCompress(nfft, thresholdVec)

    numIter = 1000
    signalIn = np.random.rand(nfft * numWindPerBlock, numIter)
    startTime = time.time()
    for i in range(numIter):
        winIdx, binIdx, binVals = sparsdrObj.work(signalIn[:, 1])
    endTime = time.time()
    print(f"Time elapsed: {endTime - startTime:.2f} s")

    megaSamplesPerSec = (signalIn.size) / (endTime - startTime) / np.power(10, 6)
    print(f"Throughput: {megaSamplesPerSec: .2f} MSps")
