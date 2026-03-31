DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
---

[![arXiv](https://img.shields.io/badge/arXiv-2602.19053-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.19053)
<!-- [![poster](https://img.shields.io/badge/NeurIPS'25|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://drive.google.com/file/d/1uh4brNIvyMsGLtoceiegJr-87K1wE_qo/view?usp=sharing) -->
<!-- [![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/YJ0HMZXnqxE) -->

<img width="1864" height="756" alt="deltaflow_cover" src="https://github.com/user-attachments/assets/a7348910-8073-4703-8c0b-57c613401552" />

## Quick Run

To train the full dataset, please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow?tab=readme-ov-file#1-data-preparation) for raw data download and h5py files preparation.

### Training

1. Prepare the **demo** train and val data for a quick run:
```bash
# around 1.3G
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo-data-v2.zip
unzip demo-data-v2.zip -d /home/kin/data/av2/h5py # to your data path
```

2. Follow the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#0-installation) to setup the environment or [use docker](https://github.com/KTH-RPL/OpenSceneFlow?tab=readme-ov-file#docker-recommended-for-isolation).

3. Run the training with the following command (modify the data path accordingly):
```bash
python train.py model=deltaflow epochs=15 batch_size=2 num_frames=5 train_aug=True \
  loss_fn=teflowLoss "voxel_size=[0.15, 0.15, 0.15]" "point_cloud_range=[-38.4, -38.4, -3, 38.4, 38.4, 3]" \
  +ssl_label=seflow_auto "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
  optimizer.name=Adam optimizer.lr=2e-3 +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=9 +optimizer.scheduler.gamma=0.5 \
  train_data=${demo_train_data_path} val_data=${demo_val_data_path}
```

### Evaluation

Here is the pretrained weights link table for different training datasets (Note that these models are only used for research and reproducibility purposes only please follow the dataset license and privacy rules to use them):

| Train Dataset | Pretrained ckpt Link |
|:--------:|:--------------:|
| Argoverse 2 | [huggingface](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/teflow/deltaflow-av2.ckpt) |
| Waymo Open Dataset | [huggingface](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/teflow/deltaflow-waymo.ckpt) |
| nuScenes | [huggingface](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/teflow/deltaflow-nus.ckpt) |

Please check the local evaluation result (raw terminal output screenshot) in [this discussion thread](TODO). 
You can also run the evaluation by yourself with the following command with trained weights:
```bash
python eval.py checkpoint=${path_to_pretrained_weights} dataset_path=${demo_data_path}
```

### Visualization

Please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#4-visualization) for visualization instructions.

## Cite & Acknowledgements

```
@inproceedings{zhang2026teflow,
  title = {{TeFlow}: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation},
  author={Zhang, Qingwen and Jiang, Chenhan and Zhu, Xiaomeng and Miao, Yunqi and Zhang, Yushan and Andersson, Olov and Jensfelt, Patric},
  year = {2026},
  booktitle = {Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages = {},
}
```
This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
