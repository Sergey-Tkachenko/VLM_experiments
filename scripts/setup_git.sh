#!/bin/bash

apt-get update && apt-get install -y ffmpeg python3-pip
git config --global user.name "serg" && git config --global user.email "dinamoares12@gmail.com"

pip install uv
uv sync