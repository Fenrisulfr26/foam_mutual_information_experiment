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


    modelNo = camera.getModelNumber();
    print("\nModelNo=" + modelNo);

    serialNo = camera.getSerialNumber();
    print("SerialNo=" + serialNo);

    fpgaSerialNo = camera.GetSerialNumber();
    print("fpgaSerialNo=" + fpgaSerialNo);


    camera.setMode(PF32_Camera.MODE_PHOTON_COUNTING);
    camera.setExposure_us(2) 
    enableFooters = True;
    camera.setEnableFooters(enableFooters)


    no_of_pixels = camera.getNoOfPixels()
    no_of_frames = 2;
    buffered = False
    perform_initial_purge = True

    print("Reading raw data")
    frames, footers, success = camera.getNextFrames(no_of_frames, buffered, perform_initial_purge)

    if not success:
        print("getNextFrames() was not successful\n")
        return

    if write_to_file:
        raw_data = open("frames.dat", "w")
    
        for f in range(0, no_of_frames):
            raw_data.write("Frame " + str(f) + "\n")
            frame_data = frames[f].get_data()
            for p in range(0, no_of_pixels):
                raw_data.write(str(frame_data[p]) + " ")
            raw_data.write("\n")

            if enableFooters:
                footer = footers[f]
                raw_data.write(str(footer.get_frame_number()) + " " + str(footer.get_x()) + " " + str(footer.get_y()) + " "+ str(footer.get_z()) + "\n")

        raw_data.write("\n")

        raw_data.close()
    else:

        raw_data_str = StringIO()

        for f in range(0, no_of_frames):
            raw_data_str.write("Frame " + str(f) + "\n")
            frame_data = frames[f].get_data()
            for p in range(0, no_of_pixels):
                raw_data_str.write(str(frame_data[p]) + " ")
            raw_data_str.write("\n")

            if enableFooters:
                footer = footers[f]
                raw_data_str.write(str(footer.get_frame_number()) + " " + str(footer.get_x()) + " " + str(footer.get_y()) + " "+ str(footer.get_z()) + "\n")

        raw_data_str.write("\n")

        print(raw_data_str.getvalue())

    factory.PF_destruct(camera)



if __name__ == '__main__':
        main()
