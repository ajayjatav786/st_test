# CodeFormer

**Paper:** [Towards Robust Blind Face Restoration with Codebook Lookup Transformer](https://arxiv.org/pdf/2206.11253.pdf)

**GitHub:** [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)

## Install

```shell script
BASE_PATH="/home/ubuntu"

# Start from scratch
cd ${BASE_PATH} &&\
rm -Rf CodeFormer

# Get sources
cd ${BASE_PATH} &&\
git clone https://github.com/sczhou/CodeFormer
    
# Remove local copy of basicsr which conflicts with the official package
cd "${BASE_PATH}/CodeFormer" &&\
rm -Rf basicsr

# Install the official basicsr (see https://github.com/xinntao/BasicSR)
pip install basicsr

# Install other dependencies 
pip install lpips gdown

# Download pre-trained models
cd "${BASE_PATH}/CodeFormer/weights/CodeFormer/" &&\
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
cd "${BASE_PATH}/CodeFormer/weights/facelib/" &&\
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth &&\
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
```

➤ Make sure that `${BASE_PATH}/CodeFormer` is in PYTHONPATH