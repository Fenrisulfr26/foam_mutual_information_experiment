#!/usr/bin/env python
import ctypes
from PF32_Factory import PF32_Factory
from PF32_Camera import PF32_Camera
from PF_Session import PF_Session
import array
from io import StringIO
import platform


def outputData(write_to_file, which_camera, data, no_of_frames, no_of_pixels):

    if write_to_file:
        file_name = "raw" + str(which_camera) + ".dat"
        raw_data = open(file_name, "w")

        for f in range(0, no_of_frames):
            raw_data.write("Frame " + str(f) + "\n")
            for p in range(0, no_of_pixels):
                raw_data.write(str(data[(f * no_of_pixels) + p]) + " ")
            raw_data.write("\n")
        raw_data.write("\n")

        raw_data.close()
    else:

        raw_data_str = StringIO()
        for f in range(0, no_of_frames):
            raw_data_str.write("Frame " + str(f) + "\n")
            for p in range(0, no_of_pixels):
                raw_data_str.write(str(data[(f * no_of_pixels) + p]) + " ")
            raw_data_str.write("\n")
        raw_data_str.write("\n")

        print(raw_data_str.getvalue())



def main():

    write_to_file = False;

    factory = PF32_Factory()
    factory.setLogStreamLevel(PF32_Factory.LOGLEVEL_TRACE)

    camera1 = factory.PF_construct()
    camera1.setExposure_us(40) 

    camera2 = factory.PF_construct()
    camera2.setMode(PF32_Camera.MODE_PHOTON_COUNTING)
    camera2.setExposure_us(ctypes.c_double(80)) 

    buffered = False
    perform_initial_purge = True

    no_of_pixels1 = camera1.getNoOfPixels()
    no_of_frames1 = 4;
    bulk_size1 = no_of_pixels1 * no_of_frames1
    data1 = (ctypes.c_uint16 * bulk_size1)() 

    no_of_pixels2 = camera2.getNoOfPixels()
    no_of_frames2 = 2;
    bulk_size2 = no_of_pixels2 * no_of_frames2
    data2 = (ctypes.c_uint16 * bulk_size2)()


    session = factory.createSession();

    session.addCamera(camera1, data1, no_of_frames1); 
    session.addCamera(camera2, data2, no_of_frames2);  

    print("Reading raw data")

    session.executeSession(buffered, perform_initial_purge);

    print("Writing raw data")

    outputData(write_to_file, 1, data1, no_of_frames1, no_of_pixels1)
    outputData(write_to_file, 2, data2, no_of_frames2, no_of_pixels2)

    factory.destroySession(session)
    factory.PF_destruct(camera1)
    factory.PF_destruct(camera2)

 


if __name__ == '__main__':
        main()
