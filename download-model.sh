#!/bin/bash


# 1. Export your Kaggle credentials
export KAGGLE_KEY=e465c7838c6af89b4fd81860726c01c9                              
export KAGGLE_USERNAME=akashrathor  

# 2. Set output path
DOWNLOAD_DIR="models/qwen"
OUTPUT_DIR="models/qwen"
MODEL_FILE="$DOWNLOAD_DIR/model.tar.gz"

#2.5  Create directories if they don’t exist
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$OUTPUT_DIR"

# Download the model
curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
  -o "$MODEL_FILE" \
  https://www.kaggle.com/api/v1/models/qwen-lm/qwen2-vl/transformers/7b-instruct/1/download

# Extract the model into the output directory
tar -xvzf "$MODEL_FILE" -C "$OUTPUT_DIR"