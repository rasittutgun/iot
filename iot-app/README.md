# IOT App

Frontend, backend and deep learning classifier (works with segmented signal outputs, binary files) part of the IOT App.\
&nbsp;
&nbsp;

# Installation

If you're not developing and will run with docker skip to **STEP Running on closed linux machine w/Docker**

Clone repository:
```
$ git clone https://USERNAME@bit.esensi.local/scm/iot/iot-app.git
```
\
Install dependencies:
```
$ cd PROJECT-DIRECTORY
$ cd angular-client
$ npm install
$ cd ../express-server
$ npm install
$ cd ../dlClassifier
$ python3 -m pip install --no-cache-dir -U -r requirements.txt
```
\
&nbsp;

# Running

There are two options for running, the first one being for development and run with node.js. Second option is for product testing on a server or closed network device using docker as the container layer.\
&nbsp;

## On Host computer (For Development)
* Before running the other components we have to pull and run a redis image for our web service. 
    ```
    $ docker pull redis #only needs to be executed once
    $ docker run -d -it --name redis-server -p 6379:6379 redis
    ```
&nbsp;

* After this step we can go to PROJECT_DIR/express-server and modify .env to run on localhost rather than the docker container:
change this:
    ```
    REDIS_HOST=redis-server #change to localhost for development
    ```
    to 
    ```
    REDIS_HOST=localhost #change to localhost for development
    ```
    OPTIONAL: Once the application is running we can open a shell on our redis container and see the rfs stored inside as per session:

    ``` 
    $ docker exec -it redis-server redis-cli
    $ keys *
    ```

&nbsp;
* Now we can run the express-server:
    ```
    $ cd PROJECT_DIR/express-server
    $ npm start
    ```
    OPTIONAL: Once the application is running we can interact with the server by calling REST endpoints on localhost:3000 using postman

&nbsp;
* Next we can run our angular frontend
    ```
    $ cd ../PROJECT_DIR/angular-client
    $ npm start
    ```
&nbsp;
* Now we can visit [localhost:4200](http://localhost:4200) and use our app.

Once finished using app be sure to remove docker containers that we brought up for this app by repeating commands for each docker container alive:

```
$ docker ps
```
In our case this returned:
```
CONTAINER ID   IMAGE     COMMAND                  CREATED       STATUS       PORTS                                       NAMES
908d449552df   redis     "docker-entrypoint.s…"   3 hours ago   Up 3 hours   0.0.0.0:6379->6379/tcp, :::6379->6379/tcp   redis-server
```
Now we can stop the container named redis-server by calling:
```
$ docker stop redis-server
$ docker rm redis-server
```

## Running w/Docker
We can use docker to build and run our app without manually going through the previous steps. However, be noted that docker will work best with linux. If you don't have docker and docker-compose installed please make sure that you do.

&nbsp;

Before building our containers we have some steps to complete:
* Go to PROJECT_DIR/express-server and modify .env to run on docker container rather than the localhost:
change this:
    ```
    REDIS_HOST=localhost #change to localhost for development
    ```
    to
    ```
    REDIS_HOST=redis-server #change to localhost for development
    ```

* Next go to PROJECT_DIR and open the docker-compose.yml file and edit to make sure that the volume that will listen for .bin file inputs coming from the signal segmentation pipeline points to the correct path, which needs to be assigned beforehand as a environmental variable on the .env file in the same directory as the docker-compose.yml file:

```
#.env
TEST_FILE_PATH=/home/baha-esen/Documents/iot-app/test
```

&nbsp;

Now we can run our application through docker by going to PROJECT_DIR and running:
```
$ docker-compose up --build
```
Optionally we can also add a -d flag if we want to run in detached mode or a --build flag if we want to ensure that images will be rebuild (use if you recently edited code).

&nbsp;

Now we can visit [localhost:4200](http://localhost:4200) and use our app.

Once finished using app be sure to remove docker containers that we brought up for this app by repeating the listed on the previous subtitle commands for each docker container alive.

&nbsp;

## Creating an executable image for closed network linux machine
Follow the instructions on run w/Docker tab ro run the full application on docker. Once application is running correctly through docker:

* See running containers on docker:
    ```
    $ docker ps
    ```
    output should contain at least these containers: 
    
<pre>CONTAINER ID   IMAGE                             COMMAND                  CREATED      STATUS         PORTS                                       NAMES
9a2e9afc1020   iotappendtoend_tf-triplet-model   &quot;python3 ./triplet_f…&quot;   3 days ago   Up 6 seconds                                               iotappendtoend_tf-triplet-model_1
5bb4f894fafc   iotappendtoend_express            &quot;docker-entrypoint.s…&quot;   3 days ago   Up 6 seconds   0.0.0.0:3000-&gt;3000/tcp, :::3000-&gt;3000/tcp   iotappendtoend_express_1
2fa6e86e9ae6   redis                             &quot;docker-entrypoint.s…&quot;   3 days ago   Up 7 seconds   0.0.0.0:6379-&gt;6379/tcp, :::6379-&gt;6379/tcp   iotappendtoend_redis-server_1
6140c6718e56   iotappendtoend_angular            &quot;docker-entrypoint.s…&quot;   5 days ago   Up 7 seconds   0.0.0.0:4200-&gt;4200/tcp, :::4200-&gt;4200/tcp   iotappendtoend_angular_1
</pre>

We observe that the running containers are named iotappendtoend_tf-triplet-model_1, iotappendtoend_express_1, iotappendtoend_redis-server_1 and iotappendtoend_angular_1 . These names will change from pc to pc so make sure to extract the names for the next step.

We will build the .tar images that will run on the closed network machine from these containers. 

```
$ docker save -o PATH/iotappendtoend_tf-triplet-model.tar iotappendtoend_tf-triplet-model_1

$ docker save -o PATH/iotappendtoend_express.tar iotappendtoend_express_1

$ docker save -o PATH/iotappendtoend_redis-server.tar iotappendtoend_redis-server_1

$ docker save -o PATH/iotappendtoend_angular.tar iotappendtoend_angular_1

```
&nbsp;

## Running app on closed network linux machine
Before running visit the closed network atlas folder (EWSNAS/bahadir.durmaz/dockerpkgs1804) and follow instructions.txt to install docker packages required for the Ubuntu 18 linux rig.

Afterwards we will transfer the saved images on the previous step to the closed network machine and load them.
```
$ docker load -i PATH/iotappendtoend_tf-triplet-model.tar
$ docker load -i PATH/iotappendtoend_express.tar
$ docker load -i PATH/iotappendtoend_redis-server.tar
$ docker load -i PATH/iotappendtoend_angular.tar
```


To run the loaded images we go to the file docker-compose.yml inside PROJECT_DIR/closedNetwork. Again before running the docker-compose refer to the steps in ```Running w/Docker``` section. Once we edit the docker-compose.yml to our specifications we can run.

```
$ docker-compose up
```
Optionally we can also add a -d flag if we want to run in detached mode or a --build flag if we want to ensure that images will be rebuild (use if you recently edited code).
&nbsp;

Now we can visit [localhost:4200](http://localhost:4200) and use our app.

Once finished using app be sure to remove docker containers that we brought up for this app by repeating the listed on the previous subtitle commands for each docker container alive.




