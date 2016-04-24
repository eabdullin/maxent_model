# A container with scientific python libraries installed
# Your analysis container should inherit this one with
# `FROM everware/science-python`

FROM everware/base

MAINTAINER Project Everware

USER root

# For python 2
RUN /bin/bash -c "source activate py27 && \
    conda install --yes numpy"


USER jupyter
WORKDIR /home/jupyter/

EXPOSE 8888