# Docker container for running my solutions to Udacity UD810 - Intro to Computer Vision code.

FROM tanmaniac/opencv3-cudagl:latest
LABEL maintainers="Tanmay Bangalore <tanmaybangalore@gmail.com>"

RUN apt-get update && apt-get install -y tmux libboost-all-dev gnuplot &&\
    rm -rf /var/lib/apt/lists/*

# Install Eigen for linear algebra
RUN cd /tmp &&\
    wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz -O Eigen.tar.gz &&\
    mkdir Eigen && tar -xf Eigen.tar.gz -C Eigen --strip-components=1 &&\
    cd Eigen && mkdir build && cd build && cmake .. && sudo make install &&\
    cd /tmp && rm -rf Eigen*

# Make a home directory so everything isn't just owned by root
ARG UID=1000
ARG GID=1000
ARG UNAME="opencv"

RUN mkdir -p /home/${UNAME} &&\
    echo "${UNAME}:x:${UID}:${GID}:${UNAME},,,:/home/${UNAME}:/bin/bash" >> /etc/passwd &&\
    echo "${UNAME}:x:${UID}:" >> /etc/group &&\
    echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${UNAME} &&\
    chmod 0440 /etc/sudoers.d/${UNAME} &&\
    chown ${UID}:${GID} -R /home/${UNAME}

USER ${UNAME}
ENV HOME /home/${UNAME}