import ctypes
import platform
import os

class PF_Session:


    def __init__(self, handle, libPF32_API):
        self.handle = handle
        self.libPF32_API = libPF32_API

        self.libPF32_API.addCamera.arg_types = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint]
        self.libPF32_API.addCamera_short.arg_types = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint]
        self.libPF32_API.executeSession.arg_types = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool]
    
    def executeSession(self, buffered, performInitialPurge):
        return self.libPF32_API.executeSession(self.handle, buffered, performInitialPurge)
    
    def addCamera(self, camera, data, noOfFrames):
        return self.libPF32_API.addCamera(self.handle, camera.handle, data, noOfFrames)
    
    def addCamera_short(self, camera, data, noOfFrames):
        return self.libPF32_API.addCamera_short(self.handle, camera.handle, data, noOfFrames)
    
