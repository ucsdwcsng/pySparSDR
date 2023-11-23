from tests.test_compressor import test_compression
from struct import pack

# 1 bit is_ave + window + time + 16 bits imag +16 bits real
def encode(output, window,bins,is_ave,bit_of_fft):
    data_number=len(output)
    time_bits  = 32-1-bit_of_fft
    file=open('binary_data.bin','wb')
    file_raw=open('int_data.txt','w')
    for i in range(data_number):
        value=output[i]
        real_part=int(value.real)
        imag_part=int(value.imag)
        index_window=window[i]
        index_bin=bins[i]
        line=["bin: ",str(index_bin)," window: ",str(index_window)," real: ",str(real_part)," imag: ",str(imag_part)]
        is_averagy=(is_ave<<31)
        index_bin_b=index_bin<<time_bits
        hdr=is_averagy+index_bin_b+index_window
        packed_data=pack('Ihh',hdr,imag_part,real_part)
        file.write(packed_data)
        file_raw.writelines(line)
        file_raw.write('\n')
    file.close()
    file_raw.close()
    return None
  
#test

winIdx,binIdx,binVals=test_compression()
encode(binVals,winIdx,binIdx,0,11)
