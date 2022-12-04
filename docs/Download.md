###### *Note: We follow the guideline provided by [MeshTransformer/docs/DOWNLOAD.md](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md)*

# Download

## Getting Started

1. Create folders that store pre-trained models and datasets.
    ```bash
    export REPO_DIR=$PWD
    mkdir -p $REPO_DIR/models  # pre-trained models
    mkdir -p $REPO_DIR/models/fastmetro_checkpoint  # model checkpoints
    mkdir -p $REPO_DIR/datasets  # datasets
    ```

2. Download HRNet-W64 pre-trained on ImageNet.

    HRNet-W64 pre-trained models can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_hrnet.sh
    ```

    The resulting data structure should follow the hierarchy as below. 
    ```
    ${REPO_DIR}  
    |-- models  
    |   |-- fastmetro_checkpoint
    |   |-- hrnet
    |   |   |-- hrnetv2_w64_imagenet_pretrained.pth
    |   |   |-- cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |-- src 
    |-- datasets 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```

3. Download model checkpoints.
    Please put the downloaded files under the `${REPO_DIR}/models/fastmetro_checkpoint` directory.

    ### Non-Parametric
    | Model                               | Dataset   | Link            |
    | ----------------------------------- | --------- | --------------- |
    | FastMETRO-S-R50                     | Human3.6M | Download (Soon) |
    | FastMETRO-L-H64                     | Human3.6M | [Download](https://drive.google.com/u/2/uc?id=1WU6q27SV7YNGCSBLypB5IGFVWMnL26io&export=download&confirm=t)        |
    | FastMETRO-S-R50                     | 3DPW      | Download (Soon) |
    | FastMETRO-L-H64                     | 3DPW      | [Download](https://drive.google.com/u/2/uc?id=19Nc-KyluAB4UmY70HoBvIRqwRFVy4jQB&export=download&confirm=t)        |
    | FastMETRO-L-H64                     | FreiHAND  | [Download](https://drive.google.com/u/2/uc?id=1u6dr0E1w15IBmstcFaihr6r-DHKFWuw1&export=download&confirm=t)        |


    ### Parametric (w/ optional SMPL parameter regressor)
    | Model           | Dataset   | Link            |
    | --------------- | --------- | --------------- |
    | FastMETRO-L-H64 | Human3.6M | Download (Soon) |
    | FastMETRO-L-H64 | 3DPW      | Download (Soon) |

    - Model checkpoints were obtained in [Conda Environment (CUDA 11.1)](../docs/Installation.md)
    - To use SMPL parameter regressor, you need to set `--use_smpl_param_regressor` as `True`

    The resulting data structure would follow the hierarchy as below. 
    ```
    ${REPO_DIR}  
    |-- models  
    |   |-- fastmetro_checkpoint
    |   |   |-- FastMETRO-L-H64_h36m_state_dict.bin
    |   |   |-- FastMETRO-L-H64_3dpw_state_dict.bin
    |   |   |-- FastMETRO-L-H64_freihand_state_dict.bin
    |   |   |-- FastMETRO-L-H64_smpl_h36m_state_dict.bin
    |   |   |-- FastMETRO-L-H64_smpl_3dpw_state_dict.bin
    |   |   |-- ...
    |   |   |-- ...
    |   |-- hrnet
    |   |   |-- hrnetv2_w64_imagenet_pretrained.pth
    |   |   |-- cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
    |-- src 
    |-- datasets 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```

4. Download SMPL and MANO models

    To run our code smoothly, please visit the following websites to download SMPL and MANO models. 

    - Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.
    - Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `${REPO_DIR}/src/modeling/data`.

    Please put the downloaded files under the `${REPO_DIR}/src/modeling/data` directory. The data structure should follow the hierarchy below. 
    ```
    ${REPO_DIR}  
    |-- models
    |-- src  
    |   |-- modeling
    |   |   |-- data
    |   |   |   |-- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    |   |   |   |-- MANO_RIGHT.pkl
    |-- datasets
    |-- README.md 
    |-- ... 
    |-- ... 
    ```
    Please check [/src/modeling/data/README.md](../src/modeling/data/README.md) for further details.


5. Download datasets and pseudo labels for training.

    We recommend to download large files with **AzCopy** for faster speed.
    AzCopy executable tools can be downloaded [here](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy). Decompress the azcopy tar file and put the executable in any path. 

    To download the annotation files, please use the following command.
    ```bash
    cd $REPO_DIR
    path/to/azcopy copy 'https://datarelease.blob.core.windows.net/metro/datasets/filename.tar' /path/to/your/folder/filename.tar
    tar xvf filename.tar  
    ```
    `filename.tar` could be `Tax-H36m-coco40k-Muco-UP-Mpii.tar`, `human3.6m.tar`, `coco_smpl.tar`, `muco.tar`, `up3d.tar`, `mpii.tar`, `3dpw.tar`, `freihand.tar`. Total file size is about 200 GB. 

    The datasets and pseudo ground truth labels are provided by [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE). We only reorganize the data format to better fit our training pipeline. We suggest to download the orignal image files from the offical dataset websites.

    The `datasets` directory structure should follow the below hierarchy.
    ```
    ${ROOT}  
    |-- models 
    |-- src
    |-- datasets  
    |   |-- Tax-H36m-coco40k-Muco-UP-Mpii  
    |   |   |-- train.yaml 
    |   |   |-- train.linelist.tsv  
    |   |   |-- train.linelist.lineidx
    |   |-- human3.6m  
    |   |   |-- train.img.tsv 
    |   |   |-- train.hw.tsv 
    |   |   |-- train.linelist.tsv    
    |   |   |-- smpl/train.label.smpl.p1.tsv
    |   |   |-- smpl/train.linelist.smpl.p1.tsv
    |   |   |-- valid.protocol2.yaml
    |   |   |-- valid_protocol2/valid.img.tsv 
    |   |   |-- valid_protocol2/valid.hw.tsv  
    |   |   |-- valid_protocol2/valid.label.tsv
    |   |   |-- valid_protocol2/valid.linelist.tsv
    |   |-- coco_smpl  
    |   |   |-- train.img.tsv  
    |   |   |-- train.hw.tsv   
    |   |   |-- smpl/train.label.tsv
    |   |   |-- smpl/train.linelist.tsv
    |   |-- muco  
    |   |   |-- train.img.tsv  
    |   |   |-- train.hw.tsv   
    |   |   |-- train.label.tsv
    |   |   |-- train.linelist.tsv
    |   |-- up3d  
    |   |   |-- trainval.img.tsv  
    |   |   |-- trainval.hw.tsv   
    |   |   |-- trainval.label.tsv
    |   |   |-- trainval.linelist.tsv
    |   |-- mpii  
    |   |   |-- train.img.tsv  
    |   |   |-- train.hw.tsv   
    |   |   |-- train.label.tsv
    |   |   |-- train.linelist.tsv
    |   |-- 3dpw 
    |   |   |-- train.img.tsv  
    |   |   |-- train.hw.tsv   
    |   |   |-- train.label.tsv
    |   |   |-- train.linelist.tsv
    |   |   |-- test_has_gender.yaml
    |   |   |-- has_gender/test.img.tsv 
    |   |   |-- has_gender/test.hw.tsv  
    |   |   |-- has_gender/test.label.tsv
    |   |   |-- has_gender/test.linelist.tsv
    |   |-- freihand
    |   |   |-- train.yaml
    |   |   |-- train.img.tsv  
    |   |   |-- train.hw.tsv   
    |   |   |-- train.label.tsv
    |   |   |-- train.linelist.tsv
    |   |   |-- test.yaml
    |   |   |-- test.img.tsv  
    |   |   |-- test.hw.tsv   
    |   |   |-- test.label.tsv
    |   |   |-- test.linelist.tsv
    |-- README.md 
    |-- ... 
    |-- ... 

    ```
