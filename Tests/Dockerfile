# Verwende das Basisimage von NVIDIA PyTorch
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Setze das Arbeitsverzeichnis im Container
WORKDIR /SegModels

RUN apt-get update && apt-get install -y git





RUN pip install segmentation-models-pytorch






COPY ~/.ssh/gitlinux /root/.ssh/id_rsa

RUN chmod 600 /root/.ssh/id_rsa

RUN echo "StrictHostKeyChecking no" > /root/.ssh/config

RUN git clone https://github.com/jan-sb/Segmentation_Models_Test






# Führe den Befehl beim Start des Containers aus
CMD ["--gpus", "all", "-it", "--name", "SegModels2"]
