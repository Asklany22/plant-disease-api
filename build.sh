#!/usr/bin/env bash
set -e

echo "Downloading model_package.zip..."
curl -L "$MODEL_URL" -o model_package.zip

echo "Done."
