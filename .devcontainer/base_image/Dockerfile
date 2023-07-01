FROM tensorflow/tensorflow:latest-gpu-jupyter

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

RUN curl -L https://github.com/alire-project/alire/releases/download/v1.2.2/alr-1.2.2-bin-x86_64-linux.zip \
        --output /tmp/alr.zip \
    && cd /tmp \
    && unzip alr.zip \
    && mv bin/alr /usr/local/bin/alr \
    && alr -n index --reset-community \
    && alr -n toolchain --select gnat_native=12.2.1 gprbuild=22.0.1 \
    && alr -n get libadalang \
    && cd $(alr get --dirname libadalang) \
    && LIBRARY_TYPE=relocatable alr -n build \
    && find . -name "*.so" -exec cp -t /usr/local/lib {} + \
    && find ~/.config/alire/cache/dependencies/$(alr get --dirname gnat_native=12.2.1)/lib/gcc/x86_64-pc-linux-gnu/12.2.0/adalib/ -name "*12.so" -exec cp -t /usr/local/lib {} + \
    && cd python \
    && pip install . \
    && cd / \
    && rm -rf /tmp/* \
    && rm -rf ~/.config/alire \
    && rm /usr/local/bin/alr

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
