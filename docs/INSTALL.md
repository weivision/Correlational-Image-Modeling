# Installation

- Download the [ImageNet](https://imagenet.stanford.edu/) dataset.

- Install `CUDA 11.3` with `cuDNN 8` following the official installation guide of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).

- Setup conda environment:
```bash
# Create environment
conda create -n cim python=3.8 -y
conda activate cim

# Install requirements
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y


# Clone CIM
git clone https://github.com/weilivision/Correlational-Image-Modeling
cd Correlational-Image-Modeling

# Install other requirements
pip install -r requirements.txt
```
