# Docker container for running my solutions to Udacity UD810 - Intro to Computer Vision code.

FROM nvidia/cuda:8.0-devel-ubuntu14.04
LABEL maintainers="Tanmay Bangalore <tanmaybangalore@gmail.com>"

RUN apt-get update && apt-get upgrade -y && apt-get install -y wget apt-utils

# Get OpenCV dependencies
RUN apt-get update && apt-get install -y build-essential checkinstall cmake pkg-config yasm git \
    gfortran libjpeg8-dev libjasper-dev libpng12-dev &&\
    apt-get install -y libtiff5-dev &&\
    apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev \
    libv4l-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev qt5-default libgtk2.0-dev \
    libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev \
    libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev \
    protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev libhdf5-dev \
    doxygen
RUN apt-get install -y python-dev python-pip python3-dev python3-pip

# Make a home directory so everything isn't just owned by root
ARG UID=1000
ARG GID=1000
ARG UNAME="opencv"

RUN mkdir -p /home/${UNAME} && \
    echo "${UNAME}:x:${UID}:${GID}:${UNAME},,,:/home/${UNAME}:/bin/bash" >> /etc/passwd && \
    echo "${UNAME}:x:${UID}:" >> /etc/group && \
    echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${UNAME} && \
    chmod 0440 /etc/sudoers.d/${UNAME} && \
    chown ${UID}:${GID} -R /home/${UNAME}

# Fetch OpenCV
RUN cd /home/${UNAME} && git clone --verbose https://github.com/opencv/opencv.git -b 2.4

RUN cd /home/${UNAME}/opencv && mkdir release && cd release && \
    cmake -G "Unix Makefiles" -DENABLE_PRECOMPILED_HEADERS=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    CMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DWITH_TBB=ON -DBUILD_NEW_PYTHON_SUPPORT=ON -DWITH_V4L=ON -DINSTALL_C_EXAMPLES=ON \
    -DINSTALL_PYTHON_EXAMPLES=ON -DBUILD_EXAMPLES=OFF -DWITH_QT=ON -DWITH_OPENGL=ON \
    -DWITH_CUDA=ON .. &&\
    make -j"$(nproc)"  && \
    make install && \
    ldconfig

USER ${UNAME}
ENV HOME /home/${UNAME}