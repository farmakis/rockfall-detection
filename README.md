<div align="center">

# 3D ML Rockfall Detection

[![python](https://img.shields.io/badge/-Python_3.6+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tensorflow](https://img.shields.io/badge/TensorFlow_2.2+-ee4c2c?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![open3d](https://img.shields.io/badge/Open3D_0.13.0+-FFFFFF?logo=open3d&logoColor=black)](https://www.open3d.org/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://opensource.org/license/mit)


Official implementation for
<br>
<br>
[_Rockfall Detection Using LiDAR and Deep Learning_](https://doi.org/10.1016/j.enggeo.2022.106836) (Engineering Geology)
<br>
[![mdpi](https://img.shields.io/badge/Elsevier-j.enggeo.2022.106836-DB7F2A.svg)](https://doi.org/10.1016/j.enggeo.2022.106836)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8042712.svg)](https://doi.org/10.5281/zenodo.8042712)
[![Project page](https://img.shields.io/badge/Project_page-8A2BE2)](https://drprojects.github.io/superpoint-transformer) -->
<br>
</div>

<br>

## 📌  Description

<p align="center">
  <img width="60%" src="./media/teaser.png">
</p>

**3D ML Rockfall Detection** is a **point cloud classification** method with [PointNet](https://arxiv.org/abs/1612.00593) and [PointNet++](https://arxiv.org/abs/1612.00593) models. The project aims at separating between rockfalls an other deformation clusters detected within rock slope change detection programs. Detailed applications of the models and analysis results on real rockfall monitoring cases are demonstrated in the associated research [paper](https://doi.org/10.1016/j.enggeo.2022.106836).
The repository includes components of the TensorFlow 2 layers provided [here](https://github.com/dgriffiths3/pointnet2-tensorflow2) and the TensorFlow operations provided [here](https://github.com/charlesq34/pointnet2/tree/master/tf_ops).

<br>

## 💻  Environment requirements
This project was tested with:
- Linux OS (Ubuntu 18.04)
- **64G** RAM
- NVIDIA GeForce RTX 2070 **8G**
- CUDA 10.1
- conda 23.3.1

<br>

## 🏗  Installation
Simply run [`install.sh`](install.sh) to install all dependencies in a new 
conda environment named `3d-ml-rd`. 
```bash
# Creates a conda env named '3d-ml-rd' env and installs dependencies
./install.sh
```

# <sub>Data preparation
The training data for the models should represent individual deformation clusters resulted from point cloud based change detection after de-noising, clustering, and meshing. Details on the data generation workflow are provided [here](https://www.mdpi.com/2220-9964/10/3/157). When both <code>rockfall</code>  and <code>non_rockfall</code> data are available in <code>.off</code> format and split into <code>train</code>, <code>dev</code>, <code>test</code> sets, copy them in the respective folders following the steps below:
  1) Create a folder called <code>data</code>
  2) In <code>data</code>, create two folders named <code>rockfall</code> and <code>non_rockfall</code>
  3) In each of the folders created in (2), create 3 folders named <code>train</code>, <code>dev</code>, and <code>test</code> and copy the <code>.off</code>       file there in

Now, you are ready to parse the data into TensorFlow records by executing:
<pre><code>python parser.py --num_points ####
</code></pre>
where <code>####</code> is the number of points to be sampled from the surface of each deformation cluster (default=500).

# <sub>Training
To train a model with the parsed data, simply run the <code>train.py</code> script with the following arguments:
  - **model:** string data type that can be either <code>pointnet</code>, <code>pointnet++_ssg</code>, or <code>pointnet++_msg</code>
  - **epochs:** integer defining the number of training epochs
  - **batch_size:** intereger defining the size of each batch of data (default=16)
  - **num_points:** integer defining the number of points sampled from each cluster's surface model and MUST be same with parser (default=500)
  - **bn:** boolean defining whether batch normalization is applied (default=True)
  - **momentum:** float defining the momentum in batch normalization (default=0.99)
  - **dropout:** float defining the keep probabilify for dropout layers (default=0.3)
  - **lr:** float defining the learning rate (default=0.001)
  - **logdir:** directory to save the trained models in a folder called <code>logs</code> (default=the selected model name)

Here is an example for training a PointNet++ model with Multi-Scale Grouping (MSG) for 100 epochs and the default settings:
  <pre><code>python train.py --model pointnet++_msg --epochs 100</code></pre>

The weights of every epoch are saved in the <code>logs</code> folder under the subfolder named by the <code>logdir</code> input argument (<code>pointnet++_msg</code> in the example above as it uses the defaults value).
  
The training logs can also be viewed by executing:
<pre><code>tensorboard --logdir=logs</code></pre>
and navigate to <code>http://localhost:6006/</code>.

# <sub>Evaluation
To evaluate the model on the test dataset, run the <<code>evaluate.py</code> script with the following arguments:
  - **model:** string data type that can be either <code>pointnet</code>, <code>pointnet++_ssg</code>, or <code>pointnet++_msg</code>
  - **epoch:** integer defining the epoch which's the trained weights will be used to make predictions
  - **batch_size:** intereger defining the size of each batch of data (default=16)
  - **num_points:** integer defining the number of points sampled from each cluster's surface model and MUST be same with parser (default=500)
  - **logdir:** directory of the saved trained models in the folder called <code>logs</code> (default=the selected model name)
  
Here is an example for evaluating the performance of the above PointNet++_MSG model on the 50th epoch:
  <pre><code>python evaluate.py --model pointnet++_msg --epoch 50</code></pre>

