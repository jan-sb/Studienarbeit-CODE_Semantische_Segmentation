# Verwende das Basisimage von NVIDIA PyTorch
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Setze das Arbeitsverzeichnis im Container
WORKDIR /SegModels

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/jan-sb/Studienarbeit-CODE_Semantische_Segmentation.git

RUN pip install segmentation-models-pytorch



# Führe den Befehl beim Start des Containers aus
CMD ["--gpus", "all", "-it", "--name", "eigseg1"]
