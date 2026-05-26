#!/usr/bin/env python
import ctypes
from PF32_Factory import PF32_Factory
from PF32_Camera import PF32_Camera
import array
from io import StringIO
import platform

def main():

    write_to_file = False;

    path_to_library = './'
    firmwareFile = "../../Firmware/PF32_USBC_Hist.bit";

    factory = PF32_Factory()
    factory.setLogStreamLevel(PF32_Factory.LOGLEVEL_TRACE)
    camera = factory.PF_constructWithCustomFirmware(firmwareFile)

    # Can also do:
    #camera = factory.PF_construct()
    #camera.loadCustomFirmware(firmwareFile)


    camera.setMode(PF32_Camera.MODE_PHOTON_COUNTING);

    camera.setExposure_us(40) 
    camera.setNoOfFramesToHistogram(100);
    camera.setNoOfBinsInHistogram(0); # 0 means all the bins available which is 1024

    print("Reading histogram data")

    no_of_histograms = 4;
    histogram = camera.getHistogramsFromFirmware(no_of_histograms)

    no_of_pixels = camera.getNoOfPixels()
    no_of_bins_in_histogram = camera.getNoOfBinsInHistogram()

    histogram_data_str = StringIO()

    for h in range(0, no_of_histograms):
        histogram_data_str.write("Histogram=" + str(h) + "\n")
        for p in range(0, no_of_pixels):
            histogram_data_str.write("Pixel=" + str(p) + "\n")
            for t in range(0, no_of_bins_in_histogram):
                histogram_data_str.write(str(histogram[(p * no_of_bins_in_histogram) + t]) + " ")
            histogram_data_str.write("\n")
        histogram_data_str.write("\n")
    histogram_data_str.write("\n")

    if write_to_file:
        histogram_data = open("fwHistogram.dat", "w");
        histogram_data.write(histogram_data_str.getvalue())
        histogram_data.close()
    else:
        print(histogram_data_str.getvalue())

    factory.PF_destruct(camera)

 


if __name__ == '__main__':
        main()
