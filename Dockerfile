FROM python:3.11-slim

# Configuring SSH
RUN apt-get update && apt-get install -y \
    python3-setuptools \
    python3-venv \
    iproute2  \
    dhcpcd5 \
    iputils-ping
       
# Install OpenSSH
RUN apt install openssh-server sudo -y
RUN apt-get update
RUN apt-get install build-essential -y

# Install git and any other dependencies
RUN apt-get update && apt-get install -y git wget libgomp1

# Create a non-root user and set up the home directory
ARG USERNAME=lmmp
ARG PASSWORD=puc-rio
ARG UID
ARG GID

RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u $UID -g $USERNAME -s /bin/bash $USERNAME && \
    echo "$USERNAME:$PASSWORD" | chpasswd && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME && \
    usermod -aG sudo $USERNAME 

# Add the user to sudoers with no password requirement for sudo
RUN echo '${USERNAME} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers    

WORKDIR /opt

# RUN python3 -m venv openpnm_virtualenv

# Install OpenPNM python package from github instead of pypi repository.
RUN git clone https://github.com/PMEAL/OpenPNM 
WORKDIR ./OpenPNM

# Install package in editable mode, i.e., 
# allows you to modify the source code of the package, 
# and the changes will be immediately reflected without needing to reinstall the package. 
# RUN /opt/openpnm_virtualenv/bin/pip install -e .
RUN pip install -e .

# RUN /opt/openpnm_virtualenv/bin/pip install -r requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/${USERNAME}/OpenPNM
# Install PoreSpy python package from github instead of pypi repository.
RUN pip install git+https://github.com/PMEAL/porespy.git@dev

# Force older version of scipy to prevent deprecated usage
# RUN /opt/openpnm_virtualenv/bin/pip install scipy==1.12.0
RUN pip install scipy==1.12.0
# RUN /opt/openpnm_virtualenv/bin/pip install pypardiso
RUN pip install pypardiso



# RUN /opt/openpnm_virtualenv/bin/pip install ipykernel
RUN pip install ipykernel

WORKDIR /home/${USERNAME}/OpenPNM

USER ${USERNAME}

# Keep the container running by starting SSH in the foreground
CMD ["/bin/sh", "-c", "echo ${PASSWORD} | sudo -S /etc/init.d/ssh start && tail -f /dev/null"]




