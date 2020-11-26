#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/nccr-itmo/FEDOT"}
PKG=${4:-"FEDOT[]"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

RAWREPO=$(echo ${REPO} | sed "s/github\.com/raw\.githubusercontent\.com/")
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir ${PKG}==${VERSION}