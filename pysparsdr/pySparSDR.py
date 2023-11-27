# /bin/python3

import numpy as np
from scipy import signal as sig
import pyfftw.interfaces.numpy_fft as fft
from struct import pack


class pySparSDRCompress:
    """
    Implementation of the SparSDR Compressor based on
    Khazraee, M., Guddeti, Y., Crow, S., Snoeren, A.C., Levchenko, K., Bharadia, D. and Schulman, A., 2019, June. Sparsdr: Sparsity-proportional backhaul and compute for sdrs. In Proceedings of the 17th Annual International Conference on Mobile Systems, Applications, and Services (pp. 391-403).
    add self.file_name which is the file name          
    add self.max_fft_size which is the maximum fft size the system support
    add self.file_handle from which we can write data in the file
    """
    def __init__(self,file_name ,max_fft_size=1024,nfft=1024, thresholdVec=None):
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

        self.max_fft_size=max_fft_size  #the maximum fft_size the system support
        # used to determine the encoded bits of fft
        self.file_name=file_name #clean all contents in the file
        self.file_handle=open(self.file_name,'ab')




    def reset(self):
        """
        Resets internal memory if the compressor needs to be re-started
        (soft-reset)
        """
        self.bufferState = 0 * self.bufferState
        self.numWinProcessed = 0
        ##clean file

    def reset_file(self):
        #clean the file and reopen it as 'ab'
        self.file_handle.close()
        self.file_handle=open(self.file_name,'wb')
        self.file_handle.close()
        self.file_handle=open(self.file_name,'ab')


    def setThreshold(self, thresholdVec):
        """
        Sets internal threshold vector
        :input: thresholdVec :shape==(1,nfft): real-valued thresholds as numpy array
        """
        assert thresholdVec.shape == (1, self.nfft)
        self.thresholdVec = thresholdVec


    def encode(self,output, window,bins,bit_of_fft=11,is_ave=0):
        data_number=len(output)
        time_bits  = 32-1-bit_of_fft
        for i in range(data_number):
            value=output[i]
            real_part=int(value.real)
            imag_part=int(value.imag)
            index_window=window[i]
            index_bin=bins[i]
            is_averagy=(is_ave<<31)
            index_bin_b=index_bin*(2**time_bits)
            hdr=is_averagy+index_bin_b+index_window
            packed_data=pack('Ihh',int(hdr),imag_part,real_part)
            self.file_handle.write(packed_data)
        return None

    def work(self, xIn):
        """
        Perform compression on input vector
        :input: xIn :numElements==k*nfft: input signal as a numpy array
        :output: the name of file that ccontains binary data

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

        self.encode(output,thresholdFlag[:, 0],thresholdFlag[:, 1],np.log2(self.max_fft_size))

        return None
