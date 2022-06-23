#!/bin/bash
docker stop $(docker ps -q --filter ancestor=signal-acquire)
exit 0