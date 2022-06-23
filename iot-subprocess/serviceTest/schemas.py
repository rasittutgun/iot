from datetime import date, datetime
from lib2to3.pgen2.token import NAME
import uuid
from ninja import Schema, Field, ModelSchema
from ninja.orm import create_schema


class SignalAcquisitionSchema(Schema):
    RECORD_FILE_DIR: str = Field(
        required=False, default="/app/data/tuncer_unknown_unprocessed")
    BURST_FILE_DIR: str = Field(
        required=False, default="/app/data/new_dataset")
    F_SAMPLING: int = Field(required=False, default=55000000)
    WIFI_BW: int = Field(required=False, default=20000000)
    PROPS_SDR: str = Field(required=False, default="Sdr\ pozisyon\ ID")
    PROPS_P_FIRST: str = Field(required=False, default="P00")
    PROPS_DEVICE: str = Field(required=False, default="Cihaz\ Pozisyon\ ID")
    PROPS_P_SECOND: str = Field(required=False, default="P01")
    PROPS_SDR_ID: str = Field(required=False, default="Sdr\ ID")
    PROPS_SDR_SECOND: str = Field(required=False, default="SDR-02")
    ACCESS_POINT_MAC: str = Field(required=False, default="74EA3AE6BF7D")
    MAC_ADDRESSES_IN_BLACK_LIST: str = Field(required=False, default="a")
    MIN_SNR: int = Field(required=False, default=5)
    IS_LIVE: str = Field(required=False, default="true")


class SignalProcessingSchema(Schema):
    REC_NUM: int = Field(required=False, default=100000)
    FREQ: int = Field(required=False, default=2462e6)
    RATE: int = Field(required=False, default=20e6)
    BW: int = Field(required=False, default=22e6)
    GAIN: int = Field(required=False, default=40)
    DUR: int = Field(required=False, default=1)
    NAME: str = Field(required=False, default="iphone_3lu")
    DIREC: str = Field(
        required=False, default="/mnt/ssd1/dockerEndtoEndDemo/ahmet_test/")
    TECH: str = Field(required=False, default="wif")
    TYPE: str = Field(required=False, default="traf")
    SLEEP: float = Field(required=False, default=0.05)
    SETUP: float = Field(required=False, default=0.05)


class ParameterSchema(SignalAcquisitionSchema, SignalProcessingSchema):
    pass


'''
class ParameterSchema(Schema):
    RECORD_FILE_DIR: str = Field(
        required=False, default="/app/data/tuncer_unknown_unprocessed")
    BURST_FILE_DIR: str = Field(
        required=False, default="/app/data/new_dataset")
    F_SAMPLING: int = Field(required=False, default=55000000)
    WIFI_BW: int = Field(required=False, default=20000000)
    PROPS_SDR: str = Field(required=False, default="Sdr\ pozisyon\ ID")
    PROPS_P_FIRST: str = Field(required=False, default="P00")
    PROPS_DEVICE: str = Field(required=False, default="Cihaz\ Pozisyon\ ID")
    PROPS_P_SECOND: str = Field(required=False, default="P01")
    PROPS_SDR_ID: str = Field(required=False, default="Sdr\ ID")
    PROPS_SDR_SECOND: str = Field(required=False, default="SDR-02")
    ACCESS_POINT_MAC: str = Field(required=False, default="74EA3AE6BF7D")
    MAC_ADDRESSES_IN_BLACK_LIST: str = Field(required=False, default="a")
    MIN_SNR: int = Field(required=False, default=5)
    IS_LIVE: str = Field(required=False, default="true")
    REC_NUM: int = Field(required=False, default=100000)
    FREQ: int = Field(required=False, default=2462e6)
    RATE: int = Field(required=False, default=20e6)
    BW: int = Field(required=False, default=22e6)
    GAIN: int = Field(required=False, default=40)
    DUR: int = Field(required=False, default=1)
    NAME: str = Field(required=False, default="iphone_3lu")
    DIREC: str = Field(
        required=False, default="/mnt/ssd1/dockerEndtoEndDemo/ahmet_test/")
    TECH: str = Field(required=False, default="wif")
    TYPE: str = Field(required=False, default="traf")
    SLEEP: float = Field(required=False, default=0.05)
    SETUP: float = Field(required=False, default=0.05)
'''
