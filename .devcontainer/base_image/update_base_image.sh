#!/bin/bash

DOCKER_BUILDKIT=1 docker build -t rowanw/ai-indent-devcontainer:latest -f Dockerfile ../..

docker push rowanw/ai-indent-devcontainer:latest
