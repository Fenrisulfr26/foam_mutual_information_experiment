import ctypes
import platform
import os

from PF32_Footer import PF32_Footer
from PF32_Frame import PF32_Frame

class PF32_Camera:

    # Enumerated types from PF_types.h

    DAC_VBD = 0                                  # Breakdown voltage DAC.
    DAC_VEB = 1                                  # Excess bias voltage DAC.
    DAC_VQ = 2                                   # Quench voltage DAC.
    DAC_VNBL = 3                                 # TDC resolution control voltage DAC.
    DAC_VIO = 4                                  # Cooled camera
    DAC_VTDC_OR_SYNC_THRESHOLD_MV = 5            # Cooled camera or USBC_V2

    STATUS_DISCONNECTED = 0          # No camera connected.
    STATUS_CONNECTED_PRE_INIT = 1    # Camera connected, but not yet initialised.
    STATUS_READY = 2                 # Camera connected, initialised, and ready for operation.
    STATUS_ERROR = 3                 # Cannot continue without human intervention, e.g. invalid firmware path provided. See error log for details.

    DATA_SOURCE_SENSOR = 0           # Real sensor image data.
    DATA_SOURCE_TEST = 1             # Test data for software debugging.

    MODE_PHOTON_COUNTING = 0      # Photon counting mode
    MODE_TCSPC_LASER_MASTER = 1   # Camera in time-resolved mode, configured to accept external laser SYNC input as TDC stop signal.
    MODE_TCSPC_SYS_MASTER = 2     # Camera in time-resolved mode, configured to generate TRIG output and TDC stop signal.
    MODE_RAW_SPAW = 10            # Raw SPAD output mode. One row of pixel SPAD signals are directly connected to each column outputs. Consult manual before using.
    MODE_TEST_PULSE_COUNTING = 11 # Counting electrical test pulses (TESTSTART signal)
    MODE_TEST_DATA_1 = 20         # Readout test pattern 1.
    MODE_TEST_DATA_2 = 21         # Readout test pattern 2. Stats at pixel 00 in the top left, increasing horizontally along the row and then vertically downwards. With each frame, all values increment by 1, wrapping around at 1023.

    MAX_SERIAL_NUMBER_LENGTH = 32
    MAX_MODEL_NUMBER_LENGTH = 32
    NO_OF_BYTES_PER_FOOTER = 16
    NO_OF_WORDS_PER_FOOTER = 8

    LOWEST_OK_ERROR_CODE = -20

    NO_OF_ROWS = 32
    NO_OF_COLUMNS = 32


    def __init__(self, handle, libPF32_API):

        self.handle = handle
        self.libPF32_API = libPF32_API

        # Do not specify setStatusCallback.arg_types. It works best without.

        self.libPF32_API.loadCustomFirmware.restype = ctypes.c_uint
        self.libPF32_API.loadCustomFirmware.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.getWidth.restype = ctypes.c_uint
        self.libPF32_API.getWidth.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getHeight.restype = ctypes.c_uint
        self.libPF32_API.getHeight.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getNoOfPixels.restype = ctypes.c_uint
        self.libPF32_API.getNoOfPixels.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getEnabledNoOfPixels.restype = ctypes.c_uint
        self.libPF32_API.getEnabledNoOfPixels.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getEnabledHeight.restype = ctypes.c_uint
        self.libPF32_API.getEnabledHeight.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getNoOfTDCCodes.restype = ctypes.c_uint
        self.libPF32_API.getNoOfTDCCodes.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getLinkStatus.restype = ctypes.c_uint
        self.libPF32_API.getLinkStatus.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setMode.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.setI2C.restype = ctypes.c_bool
        self.libPF32_API.setI2C.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libPF32_API.getI2C.restype = ctypes.c_bool
        self.libPF32_API.getI2C.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.libPF32_API.applyDACDefaultValues.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setDAC.restype = ctypes.c_bool
        self.libPF32_API.setDAC.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libPF32_API.getDAC.restype = ctypes.c_int
        self.libPF32_API.getDAC.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.getMaxValueOfDAC.restype = ctypes.c_int
        self.libPF32_API.getMaxValueOfDAC.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.getNextFrames.restype = ctypes.c_bool
        self.libPF32_API.getNextFrames.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.c_bool, ctypes.c_bool]
        self.libPF32_API.getNextFrames_short.restype = ctypes.c_bool
        self.libPF32_API.getNextFrames_short.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint, ctypes.c_bool, ctypes.c_bool]
        self.libPF32_API.getHistogram.restype = ctypes.c_bool
        self.libPF32_API.getHistogram.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_double]
        self.libPF32_API.getNoOfFramesToHistogram.restype = ctypes.c_uint
        self.libPF32_API.getNoOfFramesToHistogram.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setNoOfFramesToHistogram.arg_types = [ctypes.c_void_p, ctypes.c_uint]
        self.libPF32_API.setNoOfBinsInHistogram.arg_types = [ctypes.c_void_p, ctypes.c_uint]
        self.libPF32_API.getNoOfBinsInHistogram.restype = ctypes.c_uint
        self.libPF32_API.getNoOfBinsInHistogram.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getHistogramsFromFirmware.restype = ctypes.c_bool
        self.libPF32_API.getHistogramsFromFirmware.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint]
        self.libPF32_API.purgeHistogramFromFirmware.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getNoOfFramesInBuffer.restype = ctypes.c_uint
        self.libPF32_API.getNoOfFramesInBuffer.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setNoOfFramesInBuffer.arg_types = [ctypes.c_void_p, ctypes.c_uint]
        self.libPF32_API.getMultipleOfBuffer.restype = ctypes.c_uint
        self.libPF32_API.getMultipleOfBuffer.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setMultipleOfBuffer.arg_types = [ctypes.c_void_p, ctypes.c_uint]
        self.libPF32_API.getModelNumber.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.getSerialNumber.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.purgeBulkFrameBuffer.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getBitMode.restype = ctypes.c_uint
        self.libPF32_API.getBitMode.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getSPADEnable.restype = ctypes.c_bool
        self.libPF32_API.setSPADEnable.arg_types = [ctypes.c_void_p, ctypes.c_bool]
        self.libPF32_API.getSPADEnable.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getDataSource.restype = ctypes.c_uint
        self.libPF32_API.setDataSource.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.getDataSource.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getEXTSTOPEnable.restype = ctypes.c_bool
        self.libPF32_API.getTestPulseCount.restype = ctypes.c_int
        self.libPF32_API.getTestStartDelay.restype = ctypes.c_int
        self.libPF32_API.setEXTSTOPEnable.arg_types = [ctypes.c_void_p, ctypes.c_bool]
        self.libPF32_API.getEXTSTOPEnable.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setTestPulseCount.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.getTestPulseCount.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setTestStartDelay.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.getTestStartDelay.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setEXTSTOPDelay.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.getShutterOutState.restype = ctypes.c_bool
        self.libPF32_API.getShutterOutState.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setShutterOutState.arg_types = [ctypes.c_void_p, ctypes.c_bool]
        self.libPF32_API.getBitsPerLine.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getBitsPerLine.restype = ctypes.c_int
        self.libPF32_API.getLinesPerFrame.restype = ctypes.c_int
        self.libPF32_API.getLinesPerFrame.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getFramesToSum.restype = ctypes.c_int
        self.libPF32_API.getFramesToSum.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setFramesToSum.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.setFramePeriodAndOpticalExposure_us.arg_types = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        self.libPF32_API.getExposure_us.restype = ctypes.c_double
        self.libPF32_API.getExposure_us.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setExposure_us.arg_types = [ctypes.c_void_p, ctypes.c_double]
        self.libPF32_API.setLineTiming.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libPF32_API.getSensorClk_Hz.restype = ctypes.c_int
        self.libPF32_API.getSensorClk_Hz.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getSync_Hz.restype = ctypes.c_int
        self.libPF32_API.getSync_Hz.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getSyncDutyRatio.restype = ctypes.c_double
        self.libPF32_API.getSyncDutyRatio.arg_types = [ctypes.c_void_p]
        self.libPF32_API.getRegionsOfInterest.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool)]
        self.libPF32_API.setRegionsOfInterest.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool)]
        self.libPF32_API.GetDeviceMajorVersion.restype = ctypes.c_int
        self.libPF32_API.GetDeviceMajorVersion.arg_types = [ctypes.c_void_p]
        self.libPF32_API.GetDeviceMinorVersion.restype = ctypes.c_int
        self.libPF32_API.GetDeviceMinorVersion.arg_types = [ctypes.c_void_p]
        self.libPF32_API.GetSerialNumber.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.SetTimeout.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.UpdateWireIns.arg_types = [ctypes.c_void_p]
        self.libPF32_API.GetWireInValue.restype = ctypes.c_int
        self.libPF32_API.GetWireInValue.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        self.libPF32_API.SetWireInValue.restype = ctypes.c_int
        self.libPF32_API.SetWireInValue.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong]
        self.libPF32_API.GetWireOutValue.restype = ctypes.c_ulong
        self.libPF32_API.GetWireOutValue.arg_types = [ctypes.c_void_p, ctypes.c_int]
        self.libPF32_API.UpdateWireOuts.arg_types = [ctypes.c_void_p]
        self.libPF32_API.UpdateTriggerOuts.arg_types = [ctypes.c_void_p]
        self.libPF32_API.ActivateTriggerIn.restype = ctypes.c_int
        self.libPF32_API.ActivateTriggerIn.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self.libPF32_API.IsTriggered.restype = ctypes.c_int
        self.libPF32_API.IsTriggered.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong]
        self.libPF32_API.ReadFromPipeOut.restype = ctypes.c_long
        self.libPF32_API.ReadFromPipeOut.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.ReadFromBlockPipeOut.restype = ctypes.c_long
        self.libPF32_API.ReadFromBlockPipeOut.arg_types = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.POINTER(ctypes.c_char)]
        self.libPF32_API.getEnableFooters.restype = ctypes.c_bool
        self.libPF32_API.getActualTemp.restype = ctypes.c_double
        self.libPF32_API.getActualTemp.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setTargetTemp.arg_types = [ctypes.c_void_p, ctypes.c_double]
        self.libPF32_API.getBoardTemp.restype = ctypes.c_double
        self.libPF32_API.getBoardTemp.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setEnableCooling.restype = ctypes.c_bool
        self.libPF32_API.setEnableFooters.arg_types = [ctypes.c_void_p, ctypes.c_bool]
        self.libPF32_API.getEnableCooling.restype = ctypes.c_bool
        self.libPF32_API.getEnableCooling.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setEnableCooling.arg_types = [ctypes.c_void_p, ctypes.c_bool]
        self.libPF32_API.getEnableFooters.arg_types = [ctypes.c_void_p]
        self.libPF32_API.iteratePositionalData_short.arg_types = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint]
        self.libPF32_API.setSyncPolarity.arg_types = [ctypes.c_void_p, ctypes.c_bool]
        self.libPF32_API.getSyncThreshold.restype = ctypes.c_int
        self.libPF32_API.getSyncThreshold.arg_types = [ctypes.c_void_p]
        self.libPF32_API.setSyncThreshold.arg_types = [ctypes.c_void_p, ctypes.c_int]

    
    
    def loadCustomFirmware(self, firmwareFileName):
        b_customFirmwareFile = firmwareFileName.encode('utf-8')
        return self.libPF32_API.loadCustomFirmware(self.handle, b_customFirmwareFile)
    
    def getWidth(self):
        return self.libPF32_API.getWidth(self.handle)
    
    def getHeight(self):
        return self.libPF32_API.getHeight(self.handle)
    
    def getNoOfPixels(self):
        return self.libPF32_API.getNoOfPixels(self.handle)
    
    def getEnabledNoOfPixels(self):
        return self.libPF32_API.getEnabledNoOfPixels(self.handle)
    
    def getEnabledHeight(self):
        return self.libPF32_API.getEnabledHeight(self.handle)
    
    def getNoOfTDCCodes(self):
        return self.libPF32_API.getNoOfTDCCodes(self.handle)
    
    def getLinkStatus(self):
        return self.libPF32_API.getLinkStatus(self.handle)
    
    def setMode(self, mode):
        self.libPF32_API.setMode(self.handle, mode)
    
    def setI2C(self, data, address):
        return self.libPF32_API.setI2C(self.handle, data, address)
    
    def getI2C(self, address):
        dataOut = ctypes.c_int(8) 
        ptr = ctypes.pointer(dataOut)
        self.libPF32_API.getI2C(self.handle, address, ptr)
        return ptr.contents.value
    
    def applyDACDefaultValues(self):
        self.libPF32_API.applyDACDefaultValues(self.handle)
    
    def setDAC(self, dacType, value):
        return self.libPF32_API.setDAC(self.handle, dacType, value)
    
    def getDAC(self, dacType):
        return self.libPF32_API.getDAC(self.handle, dacType)
    
    def getMaxValueOfDAC(self, dacType):
        return self.libPF32_API.getMaxValueOfDAC(self.handle, dacType)
    

    def getNextFrames(self, no_of_frames, buffered, perform_initial_purge):
        no_of_pixels = self.getEnabledNoOfPixels()
        bulk_size = no_of_pixels * no_of_frames
        enableFooters = self.getEnableFooters()
        if enableFooters:
            bulk_size += (no_of_frames * (PF32_Camera.NO_OF_WORDS_PER_FOOTER))

        bitMode = self.getBitMode()
        data = (ctypes.c_uint16 * bulk_size)() if (bitMode == 16) else (ctypes.c_uint8 * bulk_size)()
        success = self.libPF32_API.getNextFrames(self.handle, data, no_of_frames, buffered, perform_initial_purge)

        frames = list()
        footers = list()

        if enableFooters:
            enabled_height = self.getEnabledHeight()
            for f in range(no_of_frames):
                frame_data = (ctypes.c_uint16 * no_of_pixels)() # Footers are not supported for lower bit reads
                footer_data = (ctypes.c_uint32 * PF32_Footer.NO_OF_FIELDS)()
                self.iteratePositionalData_short(data, f, frame_data, footer_data, enabled_height)
                frame = PF32_Frame(frame_data)
                frames.append(frame)
                footer = PF32_Footer(footer_data[0], footer_data[1], footer_data[2], footer_data[3])
                footers.append(footer)
        else:
            beginning_of_frame = 0
            end_of_frame = no_of_pixels

            for f in range(no_of_frames):
                frame = PF32_Frame(data[beginning_of_frame:end_of_frame])
                frames.append(frame)
                beginning_of_frame += no_of_pixels
                end_of_frame += no_of_pixels
                
        return frames, footers, success


    # For getting raw data out as quickly as possible but not as easy to use as getNextFrames()

    def getNextFrames_raw(self, data, no_of_frames, buffered, perform_initial_purge):
        return self.libPF32_API.getNextFrames_short(self.handle, data, no_of_frames, buffered, perform_initial_purge)

    
    def getHistogram(self, no_of_seconds):
        no_of_pixels = self.getNoOfPixels()
        no_of_TDC_codes = self.getNoOfTDCCodes()
        size_of_histogram = no_of_TDC_codes * no_of_pixels
        histogram = (ctypes.c_uint16 * size_of_histogram)()
        self.libPF32_API.getHistogram(self.handle, histogram, ctypes.c_double(no_of_seconds))
        return histogram
    
    def getNoOfFramesToHistogram(self):
        return self.libPF32_API.getNoOfFramesToHistogram(self.handle)

    def setNoOfFramesToHistogram(self, noOfFrames):
        return self.libPF32_API.setNoOfFramesToHistogram(self.handle, noOfFrames)

    def setNoOfBinsInHistogram(self, noOfBins):
        return self.libPF32_API.setNoOfBinsInHistogram(self.handle, noOfBins)

    def getNoOfBinsInHistogram(self):
        return self.libPF32_API.getNoOfBinsInHistogram(self.handle)

    def getHistogramsFromFirmware(self, no_of_histograms):
        no_of_pixels = self.getNoOfPixels()
        no_of_bins_in_histogram = self.getNoOfBinsInHistogram()
        size_of_histogram = no_of_bins_in_histogram * no_of_pixels
        total_size = size_of_histogram * no_of_histograms;
        histograms = (ctypes.c_uint8 * total_size)()
        self.libPF32_API.getHistogramsFromFirmware(self.handle, histograms, no_of_histograms)
        return histograms

    def purgeHistogramFromFirmware(self):
        return self.libPF32_API.purgeHistogramFromFirmware(self.handle)
    
    def getNoOfFramesInBuffer(self):
        return self.libPF32_API.getNoOfFramesInBuffer(self.handle)
    
    def setNoOfFramesInBuffer(self, noOfFrames):
        self.libPF32_API.setNoOfFramesInBuffer(self.handle, noOfFrames)
    
    def getMultipleOfBuffer(self):
        return self.libPF32_API.getMultipleOfBuffer(self.handle)
    
    def setMultipleOfBuffer(self, multiple):
        self.libPF32_API.setMultipleOfBuffer(self.handle, multiple)
    
    def getModelNumber(self):
        modelNo = ctypes.create_string_buffer(PF32_Camera.MAX_MODEL_NUMBER_LENGTH+1)
        self.libPF32_API.getModelNumber(self.handle, modelNo)
        return modelNo.value.decode('utf-8')

    
    def getSerialNumber(self):
        serialNo = ctypes.create_string_buffer(PF32_Camera.MAX_SERIAL_NUMBER_LENGTH+1)
        self.libPF32_API.getSerialNumber(self.handle, serialNo)
        return serialNo.value.decode('utf-8')
    
    def purgeBulkFrameBuffer(self):
        self.libPF32_API.purgeBulkFrameBuffer(self.handle)
    
    def getBitMode(self):
        return self.libPF32_API.getBitMode(self.handle)
    
    def setSPADEnable(self, SPAD_en):
        self.libPF32_API.setSPADEnable(self.handle, SPAD_en)
    
    def getSPADEnable(self):
        return self.libPF32_API.getSPADEnable(self.handle)
    
    def setDataSource(self, source):
        self.libPF32_API.setDataSource(self.handle, source)
    
    def getDataSource(self):
        return self.libPF32_API.getDataSource(self.handle)
    
    def setEXTSTOPEnable(self, EXTSTOP_enable):
        self.libPF32_API.setEXTSTOPEnable(self.handle, EXTSTOP_enable)
    
    def getEXTSTOPEnable(self):
        return self.libPF32_API.getEXTSTOPEnable(self.handle)
    
    def setTestPulseCount(self, testPulseCount):
        self.libPF32_API.setTestPulseCount(self.handle, testPulseCount)
    
    def getTestPulseCount(self):
        return self.libPF32_API.getTestPulseCount(self.handle)
    
    def setTestStartDelay(self, testStartDelay):
        self.libPF32_API.setTestStartDelay(self.handle, testStartDelay)
    
    def getTestStartDelay(self):
        return self.libPF32_API.getTestStartDelay(self.handle)
    
    def setEXTSTOPDelay(self, EXTSTOP_delay):
        self.libPF32_API.setEXTSTOPDelay(self.handle, EXTSTOP_delay)
    
    def setShutterOutState(self, shutterOutState):
        self.libPF32_API.setShutterOutState(self.handle, shutterOutState)
    
    def getShutterOutState(self):
        return self.libPF32_API.getShutterOutState(self.handle)
    
    def getBitsPerLine(self):
        return self.libPF32_API.getBitsPerLine(self.handle)
    
    def getLinesPerFrame(self):
        return self.libPF32_API.getLinesPerFrame(self.handle)
    
    def setFramesToSum(self, framesToSum):
        self.libPF32_API.setFramesToSum(self.handle, framesToSum)
    
    def getFramesToSum(self):
        return self.libPF32_API.getFramesToSum(self.handle)
    
    # There can be a bug with ctypes in that double parameters are not properly converted.
    # e.g.
    # PF32_Camera.setExposure_us(PF_HANDLE, 7)
    # The driver is given the value of 4.65394e-310 instead. So instead do:
    # PF32_Camera.setExposure_us(PF_HANDLE, ctypes.c_double(7))
    #
    # The only other method that accepts double params is:
    # setFramePeriodAndOpticalExposure_us()

    def setExposure_us(self, exposure):
        self.libPF32_API.setExposure_us(self.handle, ctypes.c_double(exposure))

    def setFramePeriodAndOpticalExposure_us(self, framePeriod, opticalExposure):
        self.libPF32_API.setFramePeriodAndOpticalExposure_us(self.handle, ctypes.c_double(framePeriod), ctypes.c_double(opticalExposure))

    def getExposure_us(self):
        return self.libPF32_API.getExposure_us(self.handle)
    
    def setLineTiming(self, bitsPerLine, linesPerFrame):
        self.libPF32_API.setLineTiming(self.handle, bitsPerLine, linesPerFrame)
    
    def getSensorClk_Hz(self):
        return self.libPF32_API.getSensorClk_Hz(self.handle)
    
    def getSync_Hz(self):
        return self.libPF32_API.getSync_Hz(self.handle)
    
    def getSyncDutyRatio(self):
        return self.libPF32_API.getSyncDutyRatio(self.handle)
    
    def getRegionsOfInterest(self):
        columns = (ctypes.c_bool * PF32_Camera.NO_OF_COLUMNS)()
        rows = (ctypes.c_bool * PF32_Camera.NO_OF_ROWS)()
        cPtr = ctypes.pointer(columns)
        rPtr = ctypes.pointer(rows)
        self.libPF32_API.getRegionsOfInterest(self.handle, cPtr, rPtr)
        return columns, rows
    
    def setRegionsOfInterest(self, columns, rows):
        cPtr = ctypes.pointer(columns)
        rPtr = ctypes.pointer(rows)
        self.libPF32_API.setRegionsOfInterest(self.handle, cPtr, rPtr)
    
    def GetDeviceMajorVersion(self):
        return self.libPF32_API.GetDeviceMajorVersion(self.handle)
    
    def GetDeviceMinorVersion(self):
        return self.libPF32_API.GetDeviceMinorVersion(self.handle)
    
    def GetSerialNumber(self):
        serialNo = ctypes.create_string_buffer(PF32_Camera.MAX_SERIAL_NUMBER_LENGTH+1)
        self.libPF32_API.GetSerialNumber(self.handle, serialNo)
        return serialNo.value.decode('utf-8')
    
    def SetTimeout(self, timeout):
        self.libPF32_API.SetTimeout(self.handle, timeout)
    
    def UpdateWireIns(self):
        self.libPF32_API.UpdateWireIns(self.handle)
    
    def GetWireInValue(self, epAddr):
        value = ctypes.c_int(8)
        ptr = ctypes.pointer(value)
        error_code = self.libPF32_API.GetWireInValue(self.handle, epAddr, ptr)
        return ptr.contents.value, error_code
    
    def SetWireInValue(self, ep, val, mask):
        return self.libPF32_API.SetWireInValue(self.handle, ep, val, mask)
    
    def UpdateWireOuts(self):
        self.libPF32_API.UpdateWireOuts(self.handle)
    
    def GetWireOutValue(self, epAddr):
        return self.libPF32_API.GetWireOutValue(self.handle, epAddr)
    
    def ActivateTriggerIn(self, epAddr, bit):
        return self.libPF32_API.ActivateTriggerIn(self.handle, epAddr, bit)
    
    def UpdateTriggerOuts(self):
        self.libPF32_API.UpdateTriggerOuts(self.handle)
    
    def IsTriggered(self, epAddr, mask):
        return self.libPF32_API.IsTriggered(self.handle, epAddr, mask)
    
    def ReadFromPipeOut(self, epAddr, length, data):
        return self.libPF32_API.ReadFromPipeOut(self.handle, epAddr, length, data)
    
    def ReadFromBlockPipeOut(self, epAddr, blockSize, length, data):
        return self.libPF32_API.ReadFromBlockPipeOut(self.handle, epAddr, blockSize, length, data)
    
    def setEnableFooters(self, enableFooters):
        self.libPF32_API.setEnableFooters(self.handle, enableFooters)
    
    def getEnableFooters(self):
        return self.libPF32_API.getEnableFooters(self.handle)
    
    def iteratePositionalData_short(self, data, whichFrame, frameData, positionalData, enabledHeight):
        self.libPF32_API.iteratePositionalData_short(self.handle, data, whichFrame, frameData, positionalData, enabledHeight)
    
    def setTargetTemp(self, setpoint_K):
        self.libPF32_API.setTargetTemp(self.handle, ctypes.c_double(setpoint_K))
    
    def getActualTemp(self):
        return self.libPF32_API.getActualTemp(self.handle)
    
    def getBoardTemp(self):
        return self.libPF32_API.getBoardTemp(self.handle)
    
    def setEnableCooling(self, enableCooling):
        return self.libPF32_API.setEnableCooling(self.handle, enableCooling)
    
    def getEnableCooling(self):
        return self.libPF32_API.getEnableCooling(self.handle)
    
    def setSyncPolarity(self, positive):
        return self.libPF32_API.setSyncPolarity(self.handle, positive)
    
    def setSyncThreshold(self, threshold):
        return self.libPF32_API.setSyncThreshold(self.handle, threshold)
    
    def getSyncThreshold(self):
        return self.libPF32_API.getSyncThreshold(self.handle)
    
    def createStatusCallback(self, callback):
        return ctypes.CFUNCTYPE(None, ctypes.c_int)(callback)
    
    def setStatusCallback(self, statusCallback):
        self.libPF32_API.setStatusCallback(self.handle, statusCallback)
    
    def statusMessage(self, status):
        if status == 0:
            return "Connected"
        elif status == 1:
            return "ConnectedButNotInitialised"
        elif status == 2:
            return "Ready"
        elif status == 3:
            return "Error"
        else:
            return "UnknowStatus"
    

