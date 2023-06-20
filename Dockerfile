FROM python:slim

# update some stuff
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y wget
RUN apt-get install -y git 
RUN apt-get install -y vim

# install jupyterlab
RUN pip3 install jupyterlab

## install jupyterlab
#RUN pip uninstall -y jupyterlab && \
#    pip install jupyterlab==1.0.2 && \
#    jupyter labextension install @pyviz/jupyterlab_pyviz
#
## install and enable jupyter notebook extensions
#RUN conda install -y -c conda-forge jupyter_contrib_nbextensions && \
#    jupyter contrib nbextension install --system
#
## install and enable jupyter notebook copy to clipboard functionality
#RUN jupyter nbextension install --sys-prefix https://github.com/njwhite/jupyter-clipboard/archive/master.tar.gz && \
#    jupyter nbextension enable --sys-prefix jupyter-clipboard-master/jupyter-clipboard/main

# install code    
RUN mkdir /sensorimotor
COPY . /sensorimotor
RUN chmod -R 777 /sensorimotor
RUN cd /sensorimotor && python setup.py develop

## jupyterlab exposed
#EXPOSE 8888

WORKDIR /sensorimotor

## start jupyter lab on startup
#CMD '/bin/bash'
CMD ["jupyter","lab","--ip=*","--port=8888","--no-browser","--ServerApp.iopub_data_rate_limit=10000000","--notebook-dir=/sensorimotor","--allow-root"]


# how to build
# docker build --no-cache -t satorinet/sensorimotor:v1 .
# docker push satorinet/sensorimotor:v1
# docker run --rm -it --name sensorimotor -p 8888:8888 satorinet/sensorimotor:v1
# docker run --rm -it --name sensorimotor -p 8888:8888 -v c:\repos\sensorimotor:/sensorimotor satorinet/sensorimotor:v1 jupyter lab --ip=* --port=8888 --no-browser --ServerApp.iopub_data_rate_limit=10000000 --notebook-dir=/sensorimotor --allow-root
