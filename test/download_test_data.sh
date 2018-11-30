#!/bin/bash

FILE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
IMAGE_URL="https://cdn.shopify.com/s/files/1/0362/3745/products/mockup-894996b6.jpg?v=1475529810"

# Make data folder
if [ ! -d $FILE_DIR/data ]; then
    mkdir $FILE_DIR/data
fi

# Download sample image
if [ ! -f $FILE_DIR/data/mug.jpg ]; then
    curl -s $IMAGE_URL --output $FILE_DIR/data/mug.jpg
fi
