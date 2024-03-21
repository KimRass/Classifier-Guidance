#!/bin/bash

# Check if input is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input>"
    exit 1
fi

# Get the process IDs for the input
pids=$(ps -ef | grep "$1" | grep -v grep | awk '{print $2}')

# Check if any processes were found
if [ -z "$pids" ]; then
    echo "No processes found matching input '$1'"
    exit 1
fi

# Kill the processes
echo "Killing processes with input '$1': $pids"
echo "$pids" | xargs kill -9
