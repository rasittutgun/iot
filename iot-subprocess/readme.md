# iot-subprocess
There needs to be a volume named "data" in order to make container work. Create the volume using;
``` docker volume create --name data ```

Create a folder that contains tuncer_unknown_unprocessed and new_dataset(empty folder). And put this folder's path into volumes in the docker-compose.yml file.

Put the signal-acquire.tar inside the iot-subprocess folder.

# Running With Make
``` make run ```

# Running With Docker
This command will start the Django server inside the iot-subprocess_web container 
``` docker-compose up --build ```

# Running the signal-acquire.tar
Send a POST request to ```http://127.0.0.1:8000/api/v1/sendSignalAcquisitionParameters``` along with a JSON file that contains fields from SignalAcquisitionSchema(SignalAcquisitionSchema can be seen at [127.0.0.1/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)).

# Running the record_kayit_v3_live.sh
Send a POST request to ```http://127.0.0.1:8000/api/v1/sendSignalProcessingParameters``` along with a JSON file that contains fields from SignalProcessingSchema(SignalProcessingSchema can be seen at [127.0.0.1/api/v1/docs](http://127.0.0.1:8000/api/v1/docs)).

# Stopping the signal-acquire containers
Send a GET request to ```http://127.0.0.1/api/v1/takeDownSignalAcquisition``` to stop all signal-acquire containers.
