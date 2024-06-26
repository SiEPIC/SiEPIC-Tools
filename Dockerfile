FROM quay.io/centos/centos:stream8

# CentOS 8 reached EOL Dec 31, 2021. Therefore, mirrors need to be change to the following
RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*

# Update the system and install necessary tools.
RUN dnf -y update && \
    dnf -y install wget bzip2 unzip git mesa-dri-drivers python3 python3-pip

# Install Numpy
RUN pip3 install numpy

# Install the newest version of KLayout
RUN wget https://www.klayout.org/downloads/CentOS_8/klayout-0.28.13-0.x86_64.rpm -O ~/klayout.rpm && \
    dnf -y localinstall ~/klayout.rpm && \
    rm ~/klayout.rpm

# Clone SiEPIC-Tools and SiEPIC_EBeam_PDK.
RUN mkdir -p /root/.klayout/salt && \
    cd /root/.klayout/salt && \
    git clone https://github.com/SiEPIC/SiEPIC-Tools.git && \
    git clone https://github.com/SiEPIC/SiEPIC_EBeam_PDK.git

# Set the working directory
WORKDIR /home

# Set PATH
ENV PATH="/usr/local/bin:${PATH}:/usr/local/bin/python3:/root/.local/bin"
ENV QT_QPA_PLATFORM=minimal
