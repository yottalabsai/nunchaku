#!/bin/bash

source ~/miniconda3/bin/activate image

# huggingface-cli login
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    huggingface-cli login --token $HUGGINGFACE_TOKEN
fi

exec python entrypoint/openai/api_server.py "$@"