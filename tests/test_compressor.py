import numpy as np
import time
from pysparsdr.pySparSDR import pySparSDRCompress
import cProfile
import pstats


def test_compression():
    print()
    print(f"Running compressor on random data")

    # Parameters for the compressor
    nfft = 1024
    #numWindPerBlock = 100
    numWindPerBlock = 10
    thresholdVec = 100 * np.ones((1, nfft))

    # Create a compressor object and initialize it
    sparsdrObj = pySparSDRCompress('binary_data.bin',2048,nfft, thresholdVec)
    #sparsdrObj.reset_file()

    #numIter = 1000
    numIter=100
    signalIn = np.random.rand(nfft * numWindPerBlock, numIter)
    startTime = time.time()
    for i in range(numIter):
        sparsdrObj.work(signalIn[:, 0])
    endTime = time.time()
    print(f"Time elapsed: {endTime - startTime:.2f} s")

    megaSamplesPerSec = (signalIn.size) / (endTime - startTime) / np.power(10, 6)
    print(f"Throughput: {megaSamplesPerSec: .2f} MSps")
    return None

def test_compression_profile():
    print()
    print(f"Running compressor on random data with profiling")

    profiler = cProfile.Profile()

    # Parameters for the compressor
    nfft = 1024
    numWindPerBlock = 1
    thresholdVec = 0.001 * np.ones((1, nfft))

    # Create a compressor object and initialize it
    sparsdrObj = pySparSDRCompress(nfft, thresholdVec)

    numIter = 100000
    signalIn = np.random.rand(nfft * numWindPerBlock, numIter)

    profiler.enable()
    for i in range(numIter):
        winIdx, binIdx, binVals = sparsdrObj.work(signalIn[:, 1])
    profiler.disable()

    with open("./tests/logs/sparsdr_compress.prof", "w", encoding="utf-8") as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats("tottime")
        stats.dump_stats("./tests/logs/sparsdr_compress.prof")
