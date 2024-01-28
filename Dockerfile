# Verwende das Basisimage von NVIDIA PyTorch
FROM nvcr.io/nvidia/pytorch:23.12-py3

# Setze das Arbeitsverzeichnis im Container
WORKDIR /local
COPY /testdata local/testdata

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/jan-sb/Studienarbeit-CODE_Semantische_Segmentation.git &&\
    mv /local/testdata /Studienarbeit-CODE_Semantische_Segmentation/testdata






RUN pip install segmentation-models-pytorch &&\
    pip install opencv-python &&\
    pip install opencv-contrib-python









