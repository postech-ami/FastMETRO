###### *Note: We follow the guideline provided by [MeshTransformer/docs/INSTALL.md](https://github.com/microsoft/MeshTransformer/blob/main/docs/INSTALL.md)*

# Installation

We provide two ways to install conda environments depending on CUDA versions. 
- [Conda Environment Installation (CUDA 11.1)](#cuda11.1); model checkpoints were obtained in this environment
- [Conda Environment Installation (CUDA 10.1)](#cuda10.1)

Note that [OpenDR](https://github.com/mattloper/opendr) Renderer is not compatible with recent CUDA versions.
- With CUDA 11.1, we render the 3D mesh output using [Pyrender](https://github.com/mmatl/pyrender).
- With CUDA 10.1, we render the 3D mesh output using [OpenDR](https://github.com/mattloper/opendr) or [Pyrender](https://github.com/mmatl/pyrender).

---


<a name="cuda11.1"></a>
## Conda Environment Installation (CUDA 11.1)

- Python 3.8
- Pytorch 1.8
- torchvision 0.9.0
- cuda 11.1

We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name fastmetro python=3.8
conda activate fastmetro
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install FastMETRO
git clone --recursive https://github.com/postech-ami/FastMETRO.git
cd FastMETRO
python setup.py build develop

# Install requirements
pip install -r requirements.txt

# Install manopth
pip install ./manopth/.
```

---


<a name="cuda10.1"></a>
## Conda Environment Installation (CUDA 10.1)

- Python 3.8
- Pytorch 1.4
- torchvision 0.5.0
- cuda 10.1

We suggest to create a new conda environment and install all the relevant dependencies. 

```bash
# Create a conda environment, activate the environment and install PyTorch via conda
conda create --name fastmetro python=3.8
conda activate fastmetro
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# Install OpenDR
pip install git+https://gitlab.eecs.umich.edu/ngv-python-modules/opendr.git

# Install FastMETRO
git clone --recursive https://github.com/postech-ami/FastMETRO.git
cd FastMETRO
python setup.py build develop

# Install requirements
pip install -r requirements.txt

# Install manopth
pip install ./manopth/.
```

