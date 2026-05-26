#!/usr/bin/env python
import argparse

import ctypes
import array
import numpy
import time
from PF32_Factory import PF32_Factory
from PF32_Camera import PF32_Camera
import platform

def main(frames_per_call, no_of_calls, mode, exposure, accumulations, buffer, buffer_pause, buffer_size, buffer_multiple, raw, save):

    factory = PF32_Factory()
    camera = factory.PF_construct()

    camera.setMode(mode)
    camera.setExposure_us(exposure)
    camera.setFramesToSum(accumulations)
    camera.setEnableFooters(True)

    if buffer:
        camera.setNoOfFramesInBuffer(buffer_size)
        camera.setMultipleOfBuffer(buffer_multiple)

    camera.purgeBulkFrameBuffer() # Do this last after setting up


    no_of_pixels = camera.getNoOfPixels()


    all_calls = list()

    start_time = time.time()

    if raw:
        no_of_words_per_footer = (int)(camera.NO_OF_BYTES_PER_FOOTER / 2)
        size_of_frame_and_footer = no_of_pixels + no_of_words_per_footer
        call_size = size_of_frame_and_footer * frames_per_call

        for c in range(0, no_of_calls):
           frames = (ctypes.c_uint16 * call_size)()
           camera.getNextFrames_raw(frames, frames_per_call, buffer, False)
           f = numpy.ctypeslib.as_array(frames)
           all_calls.insert(c, f)
           if buffer and buffer_pause > 0:
                 time.sleep(buffer_pause)
    else:
        for c in range(0, no_of_calls):
            frames, footers, success = camera.getNextFrames(frames_per_call, buffer, False)

            for f in range(frames_per_call):
                all_calls.append(footers[f])

            if buffer and buffer_pause > 0:
                time.sleep(buffer_pause)

    end_time = time.time()

    
    no_of_frames = frames_per_call * no_of_calls
    first_footer_encountered = False;
    no_of_dropped_frames = 0;
    previous_frame_no = 0;

    if raw:
        all_frames = numpy.zeros(0, dtype=int)
        for i in all_calls:
            all_frames = numpy.append(all_frames, i)


        for frame in range(no_of_frames):
            beginning_of_frame = frame * size_of_frame_and_footer;
            beginning_of_footer = beginning_of_frame + no_of_pixels
    
            frame_no = (all_frames[beginning_of_footer+3] << 16) + all_frames[beginning_of_footer+2]

            if first_footer_encountered:
                no_of_dropped_frames += frame_no - previous_frame_no - 1;
            else:
                first_footer_encountered = True;
    
            previous_frame_no = frame_no;

    else:
       for f in range(no_of_frames):
           footer = all_calls[f]
           frame_no = footer.get_frame_number()
       
           if first_footer_encountered:
               no_of_dropped_frames += frame_no - previous_frame_no - 1;
           else:
               first_footer_encountered = True;

           previous_frame_no = frame_no;


    no_of_seconds = end_time - start_time;
    bulk_size = no_of_pixels * no_of_frames
    megabytes = bulk_size / 1024 / 1024
    mbps = megabytes / no_of_seconds
    fps = no_of_frames / no_of_seconds;

    stats = '\nNoOfDroppedFrames=' + str(no_of_dropped_frames) + \
            '\nTotalNoOfFrames=' + str(no_of_frames) + \
            '\nSeconds=' + str(no_of_seconds) + \
            '\nMb=' + str(megabytes) +  \
            '\nmbps=' + str(mbps) +  \
            '\nfps=' + str(fps) 


    if save:
        speed_file_name = 'speed_mode-' + str(mode) + '_exp-' + str(exposure) + '_accum-' + str(accumulations) + '_framesPerCall-' + str(frames_per_call) + '_noOfCalls-' + str(no_of_calls);

        if buffer:
            speed_file_name += "_bufferSize-" + str(buffer_size) + "_bufferMultiple-" + str(buffer_multiple)
    
        speed_file_name += '.dat'
        speed_file = open(speed_file_name, "w")
        speed_file.write(stats)
        speed_file.close()
    else:
        print(stats)

    factory.PF_destruct(camera)
 


if __name__ == "__main__":

        parser = argparse.ArgumentParser(
                            prog='PF_SpeedTest',
                            description='Counts number of dropped frames for the given camera settings',
                            epilog='Photon Force Ltd')

        parser.add_argument('-f', '--framesPerCall',
                            help='How many frames you want to read from the camera for each call. Default=1024.',
                            default=1024, type=int)

        parser.add_argument('-c', '--noOfCalls',
                            help='How many calls you want to make to the camera. Frame counts will only be processed once all the calls are made and the data is collated. Default=100.',
                            default=100, type=int)

        parser.add_argument('-x', '--exposure',
                            help='The sensor exposure time for a single frame in microseconds. Default=7.',
                            default=7, type=float)

        parser.add_argument('-a', '--accumulations',
                            help='Specifies the number of sensor frames to be summed together in the camera firmware. Total exposure time per frame read by software is therefore the sensor exposure time multiplied by the number of frames to sum. Default=1',
                            default=1, type=int)

        parser.add_argument('-m', '--mode',
                            help='Sensor acquisition mode. Default=photon_counting',
                            choices = ['photon_counting', 'TCSPC_laser_master', 'TCSPC_sys_master', 'test_pulse_counting', 'test_data_2' ],
                            default='photon_counting')

        parser.add_argument('-r', '--raw',
                            help='The raw data will be returned without copying to allow for faster throughput and fewer dropped frames. Default=off',
                            action='store_true')  # on/off flag

        parser.add_argument('-s', '--save',
                            help='Output will be saved to file instead of sent to standard output. Default=off',
                            action='store_true')  # on/off flag

        parser.add_argument('-b', '--buffer',
                            help='Uses the circular buffer. Will also be set to true if any of the other buffer parameters are set. Default=off',
                            action='store_true')  # on/off flag

        parser.add_argument('-bp', '--bufferPause',
                            help='Specifies a pause in seconds after each call to emulate processing the frame data. Default=0',
                            default=argparse.SUPPRESS, type=float)

        parser.add_argument('-bs', '--bufferSize',
                            help='The size of the circular buffer. Default=4096',
                            default=argparse.SUPPRESS, type=int)

        parser.add_argument('-bm', '--bufferMultiple',
                            help='Multiple of size of buffer. Default=1 so if the data / space is available, up to the entirety of the buffer will be read out / written to each time it blocks.  2 would mean only half the buffer could be used in one go.',
                            default=argparse.SUPPRESS, type=int)


        args = parser.parse_args()

        frames_per_call = args.framesPerCall
        no_of_calls = args.noOfCalls
        mode = args.mode;

        if mode == 'photon_counting':
                camera_mode = PF32_Camera.MODE_PHOTON_COUNTING
        elif mode == 'TCSPC_laser_master':
                camera_mode = PF32_Camera.MODE_TCSPC_LASER_MASTER
        elif mode == 'TCSPC_sys_master':
                camera_mode = PF32_Camera.MODE_TCSPC_SYS_MASTER
        elif mode == 'test_pulse_counting':
                camera_mode = PF32_Camera.MODE_TEST_PULSE_COUNTING
        else: # mode == 'test_data_2'
                camera_mode = PF32_Camera.MODE_TEST_DATA_2

        exposure = args.exposure;
        if exposure < 2:
               exposure = 2;

        accumulations = args.accumulations;
        if accumulations < 1:
                accumulations = 1;

        buffer = args.buffer;
        buffer_pause = 0
        buffer_size = 4096
        buffer_multiple = 1

        if 'bufferPause' in args or 'bufferSize' in args or 'bufferMultiple' in args:
            buffer = True;

        if buffer:
            if 'bufferPause' in args:
                buffer_pause = args.bufferPause
            if 'bufferSize' in args:
                buffer_size = args.bufferSize
            if 'bufferMultiple' in args:
                buffer_multiple = args.bufferMultiple

        raw = args.raw
        save = args.save

        main(frames_per_call, no_of_calls, camera_mode, exposure, accumulations, buffer, buffer_pause, buffer_size, buffer_multiple, raw, save)



