#!/bin/bash
# Runs a script with the AI Watch faster-whisper venv and its bundled CUDA libraries.
V=~/Work/ai-watch/.venv-audio
N=$(echo $V/lib/python*/site-packages/nvidia)
LD_LIBRARY_PATH=$N/cublas/lib:$N/cudnn/lib exec $V/bin/python -I "$@"
