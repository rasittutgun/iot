from ninja import NinjaAPI
from serviceTest import services
from .schemas import SignalAcquisitionSchema, SignalProcessingSchema

api = NinjaAPI()

# This POST request takes the parameters for signal processing.
@api.post("/sendSignalProcessingParameters")
def send_parameters(request, data: SignalProcessingSchema):
    returnValue = services.sendSignalProcessingParameters(request, data)
    return returnValue

# This POST request takes the parameters for running the signal-acquire container.
@api.post("/sendSignalAcquisitionParameters")
def send_parameters(request, data: SignalAcquisitionSchema):
    returnValue = services.sendSignalAcquisitionParameters(request, data)
    return returnValue

# This GET request stops all signal-acquire containers.
@api.get("/takeDownSignalAcquisition")
def send_command(request):
    returnValue = services.takeDown(request)
    return returnValue
