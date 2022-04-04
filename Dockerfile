FROM ubuntu:latest

# Creare directory in container image for zama code 
RUN mkdir -p /user/zama

# Copy code to usr/zama in container image 
COPY . /usr/zama

# Set working directory
WORKDIR /usr/zama

# Set Env Variables
ENV FUNCTION_NUM=1
ENV DATASET_SIZE=10000
ENV HIDDEN='8,8,8'
ENV EPOCHS=200
ENV ACTIVATION='relu'

# Install requirements 
RUN apt update
RUN apt install python3-pip -y
RUN pip3 install -r requirements.txt

# Command to execute the code
CMD python3 ./main.py ${FUNCTION_NUM} ${DATASET_SIZE} ${HIDDEN} ${EPOCHS} ${ACTIVATION}
 