const express = require('express');
const cors = require('cors');
const async = require('async');

const { createClient } = require("redis");

require('dotenv').config();

const app = express()
app.use(cors({
  origin: '*'
}))

const redisSub = createClient({
  host: process.env.REDIS_HOST,
  port: process.env.REDIS_PORT
});


async function start() {
  const redisClientz = await initRedis();

  app.use(express.urlencoded({ extended: true}))

  app.listen(process.env.SERVER_PORT, () => {
    console.log(`App listening on http://localhost:${process.env.SERVER_PORT}`)
  })
}

async function initRedis() {
  redisClientz = createClient({
    host: process.env.REDIS_HOST,
    port: process.env.REDIS_PORT
  });

  redisClientz.on("error", async function(error) {
    let success = false
    while (!success) {
      try {
        redisClientz = createClient({
          host: process.env.REDIS_HOST,
          port: process.env.REDIS_PORT
        });
  
        success = true
      } catch {
        console.log('Error connecting to Redis, retrying in 1 second')
        await new Promise(resolve => setTimeout(resolve, 1000))
      }
    }
  });

  return redisClientz;
}

app.get("/flush", function(req, res) {
  redisClientz.flushall(function (err, success) {
    if (err) {
      throw new Error(err);
    }
    return res.send(success);
  });
});

app.get("/rfs", function(req, res) {
  redisClientz.keys('*rfs*', function(err, keys) {
    if (err) return console.log(err);
    if(keys){
      async.map(keys, function(key, cb) {
        redisClientz.hgetall(key, function (error,rfs) {
          if (error) return cb(error);
          cb(null, rfs);
        });
      }, function (error, results) {
            if (error) return console.log(error);
            return res.json(results);
      });
    }
  });
});

app.get('/getRfUpdates',function(req,res,next) {
  res.set({
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*"
  });

  redisSub.on('message', function(channel, message){
    var messageEvent = new ServerEvent();
    messageEvent.addData(JSON.stringify(message));
    outputSSE(req,res, messageEvent.payload())
    });

  });

function outputSSE(req, res, data) {
  res.write(data);
}

function ServerEvent() {
   this.data = "";
};

ServerEvent.prototype.addData = function(data) {
  var lines = data.split(/\n/);

  for (var i = 0; i < lines.length; i++) {
      var element = lines[i];
      this.data += "data:" + element + "\n";
  }
}

ServerEvent.prototype.payload = function() {
  var payload = "";

  payload += this.data;
  return payload + "\n";
}

start();
redisSub.subscribe(process.env.REDIS_NOTIF_CHANNEL);