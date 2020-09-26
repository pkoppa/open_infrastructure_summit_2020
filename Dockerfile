
#FROM  specifies the base image. 
FROM continuumio/anaconda2:latest



#Optional metadata to the image
LABEL maintainer="Paniraj Koppa <pkoppa@cisco.com>" \
      description="Docker for machine learning demonstration Open Infrastructure Summit 2020"



#Copies files or directories from build source and adds them to the filesystem of the container 
COPY ./ml_trial /ml_trial


#The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime.
EXPOSE 8888


#WORKDIR  specifies working directory. Any subsequent commands will assume that is the working directory.
WORKDIR /ml_trial



#The RUN instruction will execute any commands in a new layer on top of the current image and commit the results
RUN pip --no-cache-dir install \
    flask \
    flasgger==0.8.1



#The CMD provide defaults when executing a container. 
CMD python iris_swagger.py
