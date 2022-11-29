# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git ffmpeg  # libsm6 libxext6
RUN apt install nvidia-cuda-toolkit -y

COPY . .
# Install python packages
RUN pip3 install -r requirements.txt
RUN pip3 install diffusers["torch"]

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN="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"
ENV aws_id='AKIASAKJNN7D2SAHBKX4'
ENV aws_secret='LznfC7eqz8nyO4vuinJ7U04Fh5t2iDK5I7meU4Mg'

# RUN python3 download.py

EXPOSE 8000

CMD python3 -u server.py
