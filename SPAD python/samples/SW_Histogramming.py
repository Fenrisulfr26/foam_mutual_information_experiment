#!/usr/bin/env python
import ctypes
from PF32_Factory import PF32_Factory
from PF32_Camera import PF32_Camera
import array
from io import StringIO
import platform

def main():

    write_to_file = False;

    factory = PF32_Factory()
    factory.setLogStreamLevel(PF32_Factory.LOGLEVEL_TRACE)
    camera = factory.PF_construct()

    print("Reading histogram data")

    camera.setMode(PF32_Camera.MODE_PHOTON_COUNTING);
    camera.setExposure_us(100)
    camera.setFramesToSum(1) # So driver doesn't try to set TDC's beyond range

    no_of_seconds = 5
    histogram = camera.getHistogram(no_of_seconds)

    histogram_data_str = StringIO()

    no_of_pixels = camera.getNoOfPixels()
    no_of_TDC_codes = camera.getNoOfTDCCodes()

    for p in range(0, no_of_pixels):
        histogram_data_str.write("Pixel " + str(p) + "\n")
        for t in range(0, no_of_TDC_codes):
            histogram_data_str.write(str(histogram[(p * no_of_TDC_codes) + t]) + " ")
        histogram_data_str.write("\n")
    histogram_data_str.write("\n")

    if write_to_file:
        histogram_data = open("swHistogram.dat", "w");
        histogram_data.write(histogram_data_str.getvalue())
        histogram_data.close()
    else:
        print(histogram_data_str.getvalue())

    factory.PF_destruct(camera)

 


if __name__ == '__main__':
        main()
