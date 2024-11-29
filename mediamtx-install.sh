#!/usr/bin/env bash
set -e

# determine system arch
ARCH=
if [ "$(uname -m)" == 'x86_64' ]
then
    ARCH=amd64
elif [ "$(uname -m)" == 'aarch64' ]
then
    ARCH=arm64v8
elif [ "$(uname -m)" == 'i386' ] || [ "$(uname -m)" == 'i686' ]
then
    ARCH=386
else
    ARCH=arm
fi

OS=
if [ "$(uname -s)" == 'Linux' ]
then
    OS=linux
elif [ "$(uname -s)" == 'Darwin' ]
then
    echo "Program does not work on MacOS"
    exit 
fi


if [ ! $(which wget) ]; then
    echo 'Please install wget package'
    exit 1
fi

if [ ! $(which git) ]; then
    echo 'Please install git package'
    exit 1
fi

if (( $EUID != 0 )); then
    echo "Please run as root"
    exit 1
fi

if [ -z "$1" ]; then
    echo "./mediamtx-install.sh <mtx config file path>"
    exit 1
fi


VERSION="v1.9.3"
FILE_NAME="mediamtx_${VERSION}_${OS}_${ARCH}.tar.gz"
DOWNLOAD_URL="https://github.com/bluenviron/mediamtx/releases/download/$VERSION/$FILE_NAME"

mkdir -p /opt/mediamtx
cp $1 /opt/mediamtx
cp ./mediamtx.service /lib/systemd/system

cd /opt/mediamtx
echo "Downloading mediamtx for $ARCH . . ."
wget $DOWNLOAD_URL
tar xzf $FILE_NAME
rm $FILE_NAME
chmod +x mediamtx

# cp ./mediamtx.service /lib/systemd/system

systemctl enable --now mediamtx.service


