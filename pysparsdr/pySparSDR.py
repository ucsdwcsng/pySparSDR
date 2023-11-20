# /bin/python3

import numpy as np
from scipy import signal as sig
import pyfftw.interfaces.numpy_fft as fft


class pySparSDRCompress:
    """
    Implementation of the SparSDR Compressor based on
    Khazraee, M., Guddeti, Y., Crow, S., Snoeren, A.C., Levchenko, K., Bharadia, D. and Schulman, A., 2019, June. Sparsdr: Sparsity-proportional backhaul and compute for sdrs. In Proceedings of the 17th Annual International Conference on Mobile Systems, Applications, and Services (pp. 391-403).
    """

    def __init__(self, nfft=1024, thresholdVec=None):
        """
        Initialize SparSDR Compressor
        :input: nfft :shouldBeEven: Number of bins in fft
        """
        assert not nfft % 2

        self.nfft = nfft
        self.nover = int(self.nfft / 2)
        self.windowVec = sig.windows.hann(self.nfft, sym=False)
        self.windowVec = np.expand_dims(self.windowVec, axis=1)
        if thresholdVec is None:
            self.setThreshold(np.zeros((1, self.nfft)))
        else:
            self.setThreshold(thresholdVec)

        self.bufferState = np.zeros((self.nover,))
        self.numWinProcessed = 0

    def reset(self):
        """
        Resets internal memory if the compressor needs to be re-started
        (soft-reset)
        """
        self.bufferState = 0 * self.bufferState
        self.numWinProcessed = 0

    def setThreshold(self, thresholdVec):
        """
        Sets internal threshold vector
        :input: thresholdVec :shape==(1,nfft): real-valued thresholds as numpy array
        """
        assert thresholdVec.shape == (1, self.nfft)
        self.thresholdVec = thresholdVec

    def work(self, xIn):
        """
        Perform compression on input vector
        :input: xIn :numElements==k*nfft: input signal as a numpy array
        :output: (windowIdx, binIdx, binValue)
        :output: windowIdx : Index of window over all-time
        :output: binIdx : Index of bin in a particular window
        :output: binValue : Value of the binIdx at the windowIdx

        This function remembers past input and stores overlap in the bufferState
        variable
        """
        assert not xIn.size % self.nfft

        # concatenate filter state
        xIn = np.concatenate((self.bufferState, xIn))

        # Half-Overlapped windowing
        evenWindows = self.windowVec * xIn[: -self.nover].reshape((self.nfft, -1))
        oddWindows = self.windowVec * xIn[self.nover :].reshape((self.nfft, -1))

        # Fourier Transform
        evenWindows = fft.fft(evenWindows, axis=0)
        oddWindows = fft.fft(oddWindows, axis=0)

        # Interleave overlapped windows
        output = np.empty(
            (self.nfft, 2 * evenWindows.shape[1]), dtype=evenWindows.dtype
        )
        output[:, 0::2] = evenWindows
        output[:, 1::2] = oddWindows
        output = output.transpose()

        # Threshold to find areas of activity
        thresholdFlag = np.abs(output) > self.thresholdVec
        thresholdFlag = np.transpose(thresholdFlag.nonzero())

        # Select only active bins
        output = output[thresholdFlag[:, 0], thresholdFlag[:, 1]]
        thresholdFlag[:, 0] = self.numWinProcessed + thresholdFlag[:, 0]

        # Update internal states
        self.bufferState = xIn[-self.nover :]
        self.numWinProcessed = self.numWinProcessed + 2 * evenWindows.shape[1]

        return thresholdFlag[:, 0], thresholdFlag[:, 1], output
