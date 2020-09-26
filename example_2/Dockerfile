

#The FROM  specifies the base image. 
FROM python:latest




#Optional metadate to the image
LABEL maintainer="Paniraj Koppa <pkoppa@cisco.com>" \
      description="Docker for machine learning demonstration for Open Infrastructure Summit 2020"



#The RUN instruction will execute any commands in a new layer on top of the current image and commit the results.
RUN pip --no-cache-dir install \
pandas==0.24.2 \
jupyter \  
seaborn==0.9.0 \
matplotlib==3.0.3 \ 
missingno==0.4.1 \ 
numpy==1.16.3 \
sklearn




#Copies files or directories from build source  and adds them to the filesystem of the  container 
COPY titanic.ipynb titanic.csv /ml_space/



#The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime.
EXPOSE 8888



#WORKDIR  specifies working directory. Any subsequent commands will assume that is the  working directory.
WORKDIR /ml_space



#The CMD provide defaults when executing a container. 
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]

