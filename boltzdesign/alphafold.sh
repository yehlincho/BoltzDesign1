#!/bin/bash
# Get command line arguments
JSON_DIR=$1
OUTPUT_DIR=$2
GPU_NUM=$3  # New argument for GPU number
ALPHAFOLD_DIR=$4
DOCKER_NAME=$5
LOCK_DIR="$OUTPUT_DIR/locks"


# Ensure the lock directory exists
mkdir -p "$LOCK_DIR"

# Check if there are any JSON files
if ! ls "$JSON_DIR"/*.json 1> /dev/null 2>&1; then
    echo "No JSON files found in $JSON_DIR"
    exit 1
fi

# Loop through each JSON file in the directory
for json_file in "$JSON_DIR"/*.json; do
  # Extract the base name of the JSON file (without directory and extension)
  json_filename=$(basename "$json_file")
  msa_name=$(echo "$json_filename" | sed 's/.json//g' | tr '[:upper:]' '[:lower:]')

  # Check if output directory already exists for this file
  output_subdir="$OUTPUT_DIR/$msa_name"
  if [ -d "$output_subdir" ]; then
    echo "Directory $output_subdir already exists, skipping..."
    continue
  fi

  # Check if a lock file exists for this JSON file, if it does, skip this iteration
  lock_file="$LOCK_DIR/$json_filename.lock"
  if [ -f "$lock_file" ]; then
    echo "Lock file $lock_file exists, skipping $json_file..."
    continue
  fi

  # Create a lock file to indicate this file is being processed
  touch "$lock_file"

    sudo docker run \
    --volume $ALPHAFOLD_DIR:/app/alphafold \
    --volume $JSON_DIR:/root/af_input \
    --volume $OUTPUT_DIR:/root/af_output \
    --volume $ALPHAFOLD_DIR/models:/root/models \
    --volume $ALPHAFOLD_DIR/alphafold3_data_save:/root/public_databases \
    --gpus device=$GPU_NUM \
    $DOCKER_NAME \
    python /app/alphafold/run_alphafold.py \
    --json_path=/root/af_input/"$json_filename" \
    --model_dir=/root/models \
    --output_dir=/root/af_output

  # Remove the lock file after processing
  rm -f "$lock_file"
done
