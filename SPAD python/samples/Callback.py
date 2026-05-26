#!/usr/bin/env python
import ctypes
from PF32_Factory import PF32_Factory
from PF32_Camera import PF32_Camera
import array
import struct
import platform


def logCallback(verboseLevel, section, sectionLength, msg, msgLength):

    section_msg = []
    for x in range(0, sectionLength):
        asInt = section[x]
        asByte = struct.pack("B", asInt)
        section_msg.append(asByte.decode("utf-8"))

    s = (''.join(section_msg))

    message = []
    for x in range(0, msgLength):
        asInt = msg[x]
        asByte = struct.pack("B", asInt)
        message.append(asByte.decode("utf-8"))

    m = (''.join(message))
    print("Callback: [" + s + "] " + m)



def statusCallback(status):
    print("Camera status has changed")



def main():

    factory = PF32_Factory()
    camera = factory.PF_construct()

    lcb = factory.createLogCallback(logCallback)
    factory.setLogCallback(lcb)

    scb = camera.createStatusCallback(statusCallback)
    camera.setStatusCallback(scb)

    camera.setExposure_us(40) 

    factory.PF_destruct(camera)


if __name__ == '__main__':
        main()
