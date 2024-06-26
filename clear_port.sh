#!/bin/bash

PORT=8097
PID=$(lsof -ti:$PORT)

if [ ! -z "$PID" ]; then
  echo "Killing process on port $PORT with PID $PID"
  sudo kill -9 $PID
  echo "Port $PORT is now free."
else
  echo "No process is running on port $PORT."
fi