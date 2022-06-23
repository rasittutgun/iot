import subprocess
from subprocess import Popen, PIPE


#Get the data and pass it to config file.
def runSignalProcessing(aSignalProcessingSchema):
    # Get the parameters from Signal Processing Schema
    REC_NUM = aSignalProcessingSchema.REC_NUM
    FREQ = aSignalProcessingSchema.FREQ
    RATE = aSignalProcessingSchema.RATE
    BW = aSignalProcessingSchema.BW
    GAIN = aSignalProcessingSchema.GAIN
    DUR = aSignalProcessingSchema.DUR
    NAME = aSignalProcessingSchema.NAME
    DIREC = aSignalProcessingSchema.DIREC
    TECH = aSignalProcessingSchema.TECH
    TYPE = aSignalProcessingSchema.TYPE
    SLEEP = aSignalProcessingSchema.SLEEP
    SETUP = aSignalProcessingSchema.SETUP
    CHZ = "tel"
    MRK = "Apple"
    MDL = "IPhone11"
    CHR = 34
    PLG = "+"

    try: # Run the config file with Popen. 
        bashScript = Popen("./config_kayit_v3_live.sh {REC_NUM} {FREQ} {RATE} {BW} {GAIN} {DUR} {NAME} {DIREC} {TECH} {TYPE} {SLEEP} {SETUP}", stdin=PIPE, stdout=PIPE,
                           stderr=PIPE, shell=True)

    except Exception as e: #catch the exception
        print(Exception, ":", e)

# Get the data and pass it to the docker command
def runSignalAcquisition(aSchema):
    RECORD_FILE_DIR = aSchema.RECORD_FILE_DIR
    BURST_FILE_DIR = aSchema.BURST_FILE_DIR
    F_SAMPLING = aSchema.F_SAMPLING
    WIFI_BW = aSchema.WIFI_BW
    PROPS_SDR = aSchema.PROPS_SDR
    PROPS_P_FIRST = aSchema.PROPS_P_FIRST
    PROPS_DEVICE = aSchema.PROPS_DEVICE
    PROPS_P_SECOND = aSchema.PROPS_P_SECOND
    PROPS_SDR_ID = aSchema.PROPS_SDR_ID
    PROPS_SDR_SECOND = aSchema.PROPS_SDR_SECOND
    ACCESS_POINT_MAC = aSchema.ACCESS_POINT_MAC
    MAC_ADDRESSES_IN_BLACK_LIST = aSchema.MAC_ADDRESSES_IN_BLACK_LIST
    MIN_SNR = aSchema.MIN_SNR
    IS_LIVE = aSchema.IS_LIVE

    try:
        # Don't forget the set the paths of tuncer_unknown_unprocessed and new_dataset according to your host path.
        # Just change the path before the colon(:), do NOT change the path after the colon
        # Example /home/Desktop/tuncer_unknown_unprocessed/:/app/data/tuncer_unknown_unprocessed

        #Image name of the main container
        IMAGE_NAME_OF_THE_MAIN_CONTAINER = "iot-subprocess_web"

        bashScript = Popen(
            f'docker run --rm -e "DISPLAY=: 1" -v /tmp/.X11-unix:/tmp/.X11-unix --volumes-from $(docker ps -q --filter ancestor={IMAGE_NAME_OF_THE_MAIN_CONTAINER}) -e RECORD_FILE_DIR={RECORD_FILE_DIR} -e BURST_FILE_DIR={BURST_FILE_DIR} -e F_SAMPLING={F_SAMPLING} -e WIFI_BW={WIFI_BW} -e PROPS_SDR={PROPS_SDR} -e PROPS_P_FIRST={PROPS_P_FIRST} -e PROPS_DEVICE={PROPS_DEVICE} -e PROPS_P_SECOND={PROPS_P_SECOND} -e PROPS_SDR_ID={PROPS_SDR_ID} -e PROPS_SDR_SECOND={PROPS_SDR_SECOND} -e ACCESS_POINT_MAC={ACCESS_POINT_MAC} -e MAC_ADDRESSES_IN_BLACK_LIST={MAC_ADDRESSES_IN_BLACK_LIST} -e MIN_SNR={MIN_SNR} -e IS_LIVE={IS_LIVE} signal-acquire', stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)

    except Exception as e:
        print(Exception, ":", e)
        print("probably process already exited, no problem.")


# Take down all of the the signal-acquire containers.
def takeDownSignalAcquisition():
    try:
        # Run the docker stop command
        bashScript = Popen(
            "docker stop $(docker ps -q --filter ancestor=signal-acquire)", shell=True)
        print("Succesfully stopped.")
    except:
        print("Couldn't stop the container.")