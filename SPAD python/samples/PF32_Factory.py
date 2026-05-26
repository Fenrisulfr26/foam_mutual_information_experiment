import ctypes
import platform
import os
from PF32_Camera import PF32_Camera
from PF_Session import PF_Session

class PF32_Factory:

    # Enumerated types from PF_types.h

    LOGLEVEL_TRACE = 1     # Low level execution tracing information.
    LOGLEVEL_DEBUG = 2     # Debugging information
    LOGLEVEL_INFO  = 3     # Status information.
    LOGLEVEL_WARNING  = 4  # Warnings.
    LOGLEVEL_ERROR = 5     # Critical errors.
    LOGLEVEL_OFF   = 6     # Error logging disabled.

    MAX_LOG_LEVEL = LOGLEVEL_OFF;

    library_is_loaded = False;


    def __init__(self, path_to_library = '', customFirmwareFile = None):
     
        if path_to_library == '':
            sys_name = platform.system();

            if sys_name == 'Windows':
                path_to_library = os.path.dirname(os.path.abspath(__file__))
            elif sys_name == 'Linux':
                path_to_library = './'
            else:
                print('Error: Unsupported platform (assuming Linux)')
                path_to_library = './'

        if not PF32_Factory.library_is_loaded:
            self.loadLibrary(path_to_library)
            PF32_Factory.shared_libPF32_API = self.libPF32_API
        else:
            self.libPF32_API = PF32_Factory.shared_libPF32_API


    
    def loadLibrary(self, path_to_library):
        PF32_Factory.library_is_loaded = True

        if not path_to_library[-1] == os.sep:
            path_to_library += os.sep

        sys_name = platform.system();
 
        if sys_name == 'Windows':
            self.libPF32_API = ctypes.cdll.LoadLibrary(path_to_library + 'PF32_API.dll')
        elif sys_name == 'Linux':
            self.libPF32_API = ctypes.cdll.LoadLibrary(path_to_library + 'libPF32_API.so')
        else:
            print('Error: Unsupported platform (assuming Linux)')
            self.libPF32_API = ctypes.cdll.LoadLibrary(path_to_library + 'libPF32_API.so')

        # Do not specify setLogCallback.arg_types. It works best without. 

        self.libPF32_API.getVersionMajor.restype = ctypes.c_uint
        self.libPF32_API.getVersionMinor.restype = ctypes.c_uint
        self.libPF32_API.getVersionPatch.restype = ctypes.c_uint
        self.libPF32_API.noOfPF32sInstantiated.restype = ctypes.c_int
        self.libPF32_API.getPF32InstanceByIndex.restype = ctypes.POINTER(ctypes.c_void_p)
        self.libPF32_API.getPF32InstanceByIndex.arg_types = [ctypes.c_int]
        self.libPF32_API.createSession.restype = ctypes.POINTER(ctypes.c_void_p)
        self.libPF32_API.destroySession.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getLogFileLevel.restype = ctypes.c_int
        self.libPF32_API.setLogFileLevel.arg_types = [ctypes.c_int]
        self.libPF32_API.getLogStreamLevel.restype = ctypes.c_int
        self.libPF32_API.setLogStreamLevel.arg_types = [ctypes.c_int]
        self.libPF32_API.PF32_construct.restype = ctypes.POINTER(ctypes.c_void_p)
        self.libPF32_API.PF32_constructWithCustomFirmware.restype = ctypes.POINTER(ctypes.c_void_p)
        self.libPF32_API.PF32_constructWithCustomFirmware.arg_types = [ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.PF32_destruct.arg_types = [ctypes.c_void_p]


    def getVersionMajor(self):
        return self.libPF32_API.getVersionMajor()
    
    def getVersionMinor(self):
        return self.libPF32_API.getVersionMinor()
    
    def getVersionPatch(self):
        return self.libPF32_API.getVersionPatch()
    
    def closeAll(self):
        self.libPF32_API.closeAll()
    
    def noOfPFsInstantiated(self):
        return self.libPF32_API.noOfPF32sInstantiated()
    
    def getPFInstanceByIndex(self, index):
        return self.libPF32_API.getPF32InstanceByIndex(index)
    
    def createSession(self):
        handle = self.libPF32_API.createSession()
        return PF_Session(handle, self.libPF32_API)

    def destroySession(self, session):
        self.libPF32_API.destroySession(session.handle)
    
    def setLogFileLevel(self, newLogLevel):
        self.libPF32_API.setLogFileLevel(newLogLevel)
    
    def getLogFileLevel(self):
        return self.libPF32_API.getLogFileLevel()
    
    def setLogStreamLevel(self, newLogLevel):
        self.libPF32_API.setLogStreamLevel(newLogLevel)
    
    def getLogStreamLevel(self):
        return self.libPF32_API.getLogStreamLevel()

    def createLogCallback(self, callback):
        return ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int)(callback)

    def setLogCallback(self, logCallback):
        self.libPF32_API.setLogCallback(logCallback)
    
    def PF_construct(self):
        handle = self.libPF32_API.PF32_construct()
        return PF32_Camera(handle, self.libPF32_API)
    
    def PF_constructWithCustomFirmware(self, firmwareFileName):
        b_customFirmwareFile = firmwareFileName.encode('utf-8');
        handle = self.libPF32_API.PF32_constructWithCustomFirmware(b_customFirmwareFile)
        return PF32_Camera(handle, self.libPF32_API)
    
    def PF_destruct(self, camera):
        self.libPF32_API.PF32_destruct(camera.handle)
    
    
