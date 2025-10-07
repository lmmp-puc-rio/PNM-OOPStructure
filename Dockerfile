FROM astral/uv:python3.11-bookworm-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    iproute2  \
    dhcpcd5 \
    iputils-ping \
    openssh-server \
    sudo \
&& rm -rf /var/lib/apt/lists/*

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

WORKDIR /opt/OpenPNM

# Install package in editable mode, i.e., 
# allows you to modify the source code of the package, 
# and the changes will be immediately reflected without needing to reinstall the package. 
RUN uv sync

WORKDIR /home/${USERNAME}/OpenPNM

USER ${USERNAME}

# Keep the container running by starting SSH in the foreground
CMD ["/bin/sh", "-c", "echo ${PASSWORD} | sudo -S /etc/init.d/ssh start && tail -f /dev/null"]
