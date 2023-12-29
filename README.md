# PointVST: Self-Supervised Pre-training for 3D Point Clouds via View-Specific Point-to-Image Translation

This is the official implementation of **[[PointVST](https://arxiv.org/pdf/2212.14197.pdf)] (TVCG 2023)**, a self-supervised learning approach for pre-training deep 3D point cloud backbone encoders.

<p align="center"> <img src="https://github.com/keeganhk/PointVST/blob/master/sketch.png" width="45%"> </p>

<p align="center"> <img src="https://github.com/keeganhk/PointVST/blob/master/workflow.png" width="90%"> </p>


Given a backbone point cloud encoder, we pre-train its model parameters via our proposed pretext task of *view-specific point-to-image translation* (refer to the demo scripts provided in ```main/backbone_pretraining```). The pre-trained backbones can be integrated into task-specific learning frameworks for downstream task scenarios (refer to the demo scripts provided in ```main/example_downstream```). The used datasets can be downloaded from [here](https://drive.google.com/drive/folders/1YDMVOIJbYF3YbPCY007i5phkIi_l0elG?usp=drive_link). The checkpoints of pre-trained backbone network parameters as well as various task-specific learning frameworks can also be downloaded from [here](https://drive.google.com/drive/folders/1f5NE_BuxvNFzY3dJFZdetX3cDHCTMrbU?usp=sharing).



### Citation
If you find our work useful in your research, please consider citing:

	@article{zhang2023pointvst,
	  title={PointVST: Self-Supervised Pre-Training for 3D Point Clouds Via View-Specific Point-to-Image Translation},
	  author={Zhang, Qijian and Hou, Junhui},
	  journal={IEEE Transactions on Visualization and Computer Graphics},
	  year={2023}
	}

