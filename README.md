# [ECCV'22] FastMETRO
### [Paper](https://github.com/postech-ami/FastMETRO) | [Project Page](https://github.com/postech-ami/FastMETRO) 

- This is the official PyTorch implementation of [Cross-Attention of Disentangled Modalities for 3D Human Mesh Recovery with Transformers](https://github.com/postech-ami/FastMETRO).
- **FastMETRO** (**Fast** **ME**sh **TR**ansf**O**rmer) has a novel transformer encoder-decoder architecture for 3D human pose and mesh reconstruction from a single RGB image.

---

<p align="center">  
<img src="./assets/visualization_attentions.png">  
</p> 

Our code is under refactoring. In the meantime, please refer to [this code snippet](https://github.com/postech-ami/FastMETRO) which contains the main framework of our FastMETRO.

## Citation
If you find our work useful for your research, please consider citing our paper:

````BibTeX
@InProceedings{cho2022FastMETRO,
    title={Cross-Attention of Disentangled Modalities for 3D Human Mesh Recovery with Transformers},
    author={Junhyeong Cho and Kim Youwang and Tae-Hyun Oh},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
````

## Acknowledgements
Our repository is modified and adapted from these amazing repositories. If you find their work useful for your research, please also consider citing them.
- [METRO](https://github.com/microsoft/MeshTransformer)          
- [MeshGraphormer](https://github.com/microsoft/MeshGraphormer)
- [GraphCMR](https://github.com/nkolot/GraphCMR)
- [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE)
- [HMR](https://github.com/akanazawa/hmr)
- [DETR](https://github.com/facebookresearch/detr)

## Contact
jhcho99.cs@gmail.com

## License
This research code is released under the MIT license. Please see [LICENSE](./LICENSE) for more information.

---
###### *This work was done @ POSTECH Algorithmic Machine Intelligence Lab*