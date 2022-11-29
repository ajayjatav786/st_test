# GFPGAN

**GitHub:** [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)

## Install

```shell script
BASE_PATH="/home/ubuntu"

# Get sources
cd ${BASE_PATH} &&\
git clone https://github.com/TencentARC/GFPGAN.git


# See https://github.com/xinntao/BasicSR
pip install basicsr

# See https://github.com/xinntao/facexlib
pip install facexlib

# Get pre-trained models
cd "${BASE_PATH}/models" &&\
mkdir GFPGAN &&\
cd GFPGAN &&\
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth &&\
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth &&\
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/RestoreFormer.pth &&\
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth &&\
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth
```

➤ Make sure that `/home/ubuntu/GFPGAN` is in PYTHONPATH