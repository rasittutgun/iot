from requests import request
from .schemas import SignalAcquisitionSchema, SignalProcessingSchema
from serviceTest import subprocessTest

# This method calls the config_kayit_v3_live.sh
def sendSignalProcessingParameters(request, data):
    aSchema = SignalProcessingSchema(REC_NUM=data.REC_NUM, FREQ=data.FREQ, RATE=data.RATE, BW=data.BW, GAIN=data.GAIN,
                                     DUR=data.DUR, NAME=data.NAME, DIREC=data.DIREC, TECH=data.TECH, TYPE=data.TYPE, SLEEP=data.SLEEP, SETUP=data.SETUP)
    subprocessTest.runSignalProcessing(aSchema)
    return aSchema

# This method calls the function that starts the signal-acquire container.
def sendSignalAcquisitionParameters(request, data):
    aSchema = SignalAcquisitionSchema(RECORD_FILE_DIR=data.RECORD_FILE_DIR, BURST_FILE_DIR=data.BURST_FILE_DIR, F_SAMPLING=data.F_SAMPLING, WIFI_BW=data.WIFI_BW, PROPS_SDR=data.PROPS_SDR, PROPS_P_FIRST=data.PROPS_P_FIRST, PROPS_DEVICE=data.PROPS_DEVICE,
                                      PROPS_P_SECOND=data.PROPS_P_SECOND, PROPS_SDR_ID=data.PROPS_SDR_ID, PROPS_SDR_SECOND=data.PROPS_SDR_SECOND, ACCESS_POINT_MAC=data.ACCESS_POINT_MAC, MAC_ADDRESSES_IN_BLACK_LIST=data.MAC_ADDRESSES_IN_BLACK_LIST, MIN_SNR=data.MIN_SNR, IS_LIVE=data.IS_LIVE)
    subprocessTest.runSignalAcquisition(aSchema)
    return aSchema

# This method takes down all of the signal-acquire containers.
def takeDown(request):
    return subprocessTest.takeDownSignalAcquisition()
