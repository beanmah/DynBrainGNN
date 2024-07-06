# DynBrainGNN

<p align="left">
<a href="https://github.com/beanmah/DynBrainGNN/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-MIT-blue" /></a>
</p>


Official implementation of "DynBrainGNN: Towards Spatio-Temporal Interpretable Graph Neural Network Based on Dynamic Brain Connectome for Psychiatric Diagnosis". [[paper](https://link.springer.com/chapter/10.1007/978-3-031-45676-3_17)]


## Code Structure
```plaintext
DynBrainGNN
├── data
|   ├── ABIDE
|   ├── MDD
|   └── SRPBS
├── models
|   ├── gcn.py
|   ├── spatial_attention_net.py
|   ├── temporal_attention_net.py
|   └── vae.py
├── main_ddp.py
└── utils.py
```


## Experiments
To train a DynBrainGNN model, run:
```
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --master_port 1235 main_ddp.py
```
### Command Explanation
- `nnodes`:  **Number of nodes (machines) to use.**
- `nproc_per_node`:  **Number of processes per node (typically the number of GPUs per node).**
- `master_port`:  **Port for inter-process communication.** 

This command sets up distributed training using PyTorch's distributed launcher. Adjust `nnodes` and `nproc_per_node` based on your hardware configuration.


## Citation
If you find this codebase/paper useful for your research, please consider citing:
```
@inproceedings{zheng2023dynbraingnn,
  title={DynBrainGNN: Towards Spatio-Temporal Interpretable Graph Neural Network Based on Dynamic Brain Connectome for Psychiatric Diagnosis},
  author={Zheng, Kaizhong and Ma, Bin and Chen, Badong},
  booktitle={International Workshop on Machine Learning in Medical Imaging},
  pages={164--173},
  year={2023},
  organization={Springer}
}
```