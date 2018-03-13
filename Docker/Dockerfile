# Docker container for running my solutions to Udacity UD810 - Intro to Computer Vision code.

FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainers="Tanmay Bangalore <tanmaybangalore@gmail.com>"

RUN apt-get update && apt-get upgrade -y && apt-get install -y sudo build-essential cmake git \
    clang-format wget
RUN apt-get install -y --no-install-recommends cuda-samples-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

# Get OpenCV dependencies
RUN apt-get update && apt-get install -y libgtk2.0-dev pkg-config python-numpy python-dev libavcodec-dev \
    libavformat-dev libswscale-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev \
    checkinstall pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev \
    libavformat-dev libswscale-dev libdc1394-22-dev libxine2 libgstreamer0.10-dev libv4l-dev \
    libgstreamer-plugins-base0.10-dev python-dev python-numpy libtbb-dev libgtk2.0-dev \
    libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev \
    libxvidcore-dev x264 v4l-utils
RUN apt-get install -y libqt5gui5 && apt-get install -y qtbase5-dev && apt-get install -y qt5-default

# Make a home directory so everything isn't just owned by root
ARG UID=1000
ARG GID=1000
ARG UNAME="introcv"

RUN mkdir -p /home/${UNAME} && \
    echo "${UNAME}:x:${UID}:${GID}:${UNAME},,,:/home/${UNAME}:/bin/bash" >> /etc/passwd && \
    echo "${UNAME}:x:${UID}:" >> /etc/group && \
    echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${UNAME} && \
    chmod 0440 /etc/sudoers.d/${UNAME} && \
    chown ${UID}:${GID} -R /home/${UNAME}

# Fetch OpenCV
#RUN cd /home/${UNAME} && \
#    wget https://github.com/opencv/opencv/archive/2.4.13.6.tar.gz -O opencv-2.4.13.6.tar.gz && \
#    tar -xf opencv-2.4.13.6.tar.gz
RUN cd /home/${UNAME} && git clone --verbose https://github.com/opencv/opencv.git -b 2.4

RUN cd /home/${UNAME}/opencv && mkdir release && cd release && \
    cmake -G "Unix Makefiles" -DENABLE_PRECOMPILED_HEADERS=OFF -DCMAKE_CXX_COMPILER=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_TBB=ON \
    -DBUILD_NEW_PYTHON_SUPPORT=ON -DWITH_V4L=ON -DINSTALL_C_EXAMPLES=ON \
    -DINSTALL_PYTHON_EXAMPLES=ON -DBUILD_EXAMPLES=OFF -DWITH_QT=ON -DWITH_OPENGL=ON \
    -DWITH_CUDA=ON .. &&\
    make -j"$(nproc)"  && \
    make install && \
    ldconfig

USER ${UNAME}
ENV HOME /home/${UNAME}