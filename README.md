# GhostingNet: a Novel Approach for Glass Surface Detection with Ghosting Cues

**Tao Yan, Jiahui Gao, Ke Xu, Helong Li, Xiangjie Zhu, Hao Huang, Benjamin Wah\* and Rynson W.H. Lau\*, "GhostingNet: a Novel Approach for Glass Surface Detection with Ghosting Cues," IEEE Transactions on Pattern Analysis andMachine Inteligence, September 2025**

> Abstract—Ghosting effects typically appear on glass surfaces, as each piece of glass has two contact surfaces causing two slightly offset layers of reflections. In this paper, we propose to take advantage of this intrinsic property of glass surfaces and apply it to glass surface detection, with two main technical novelties. First, we formulate a ghosting image formation model to describe the intensity and spatial relations among the main reflections and the background transmission within the glass region. Based on this model, we construct a new Glass Surface Ghosting Dataset (GSGD) to facilitate glass surface detection, with ∼3.7K glass images and corresponding ghosting masks and glass surface masks. Second, we propose a novel method, called GhostingNet, for glass surface detection. Our method consists of a Ghosting Effects Detection (GED) module and a Ghosting Surface Detection (GSD) module. The key component of our GED module is a novel Double Reflection Estimation (DRE) block that models the spatial offsets of reflection layers for ghosting effect detection. The detected ghosting effects are then used to guide the GSD module for glass surface detection. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods. We will release our code and dataset.

### Metrics Comparison

| *Metric*                          | ***use mask*** | *use boundary* |   *use extra clues*   |             *main method*             |                *main idea*                 |
| --------------------------------- | :------------: | :------------: | :-------------------: | :-----------------------------------: | :----------------------------------------: |
| **PSPNet**                        |       ✔        |       ✖        |           ✖           |        spatial pyramid pooling        |         different receptive fileds         |
| **SCWSSOD**                       |       ✖        |       ✖        |  ✔ (scribble labels)  | adaptive loss + feature normalization |            multiscale features             |
| **MINet**                         |       ✔        |       ✖        |           ✖           |           dense connection            |       feature interaction and fusion       |
| **GDNet**                         |       ✔        |       ✖        |           ✖           |            dialation conv             |         different receptive fileds         |
| **TransLab**                      |       ✔        |       ✔        |           ✖           |  dialation conv + boundary attention  | different receptive fileds + boundary cues |
| **EBLNet**                        |       ✔        |       ✔        |           ✖           |       cascade net + graph conv        |               boundary cues                |
| **GSDNet**                        |       ✔        |       ✖        | ✔ (single refletion)  | dialation conv + reflection detection |  different receptive fileds + prior cues   |
| <font color='red'>**Ours**</font> |       ✔        |       ✖        | ✔ (double reflection) |      double reflection detection      |            prior intrinsic cues            |

### Network Architecture

![NetworkArchitecture](https://github.com/YT3DVision/GhostingNet/blob/main/utils/GhostingNet.png)

### Quantitative Comparisons

![QuantitativeComparisons](https://github.com/YT3DVision/GhostingNet/blob/main/utils/QuantitativeComparisons.png)

### Qualitative comparisons

<p align="center">
  <img src="https://github.com/YT3DVision/GhostingNet/blob/main/utils/compare_results.jpg" style="width:100%" />
</p>

### Installation

The implementation of DCN based on [DCNv2](https://github.com/JBlackRainZ/DCNv2), and there are two version of DCNv2 on Linux and Windows platform respectively. If you have any problem of installing DCNv2 on Windows platform, please refer to [here](https://github.com/YT3DVision/GhostingNet/blob/main/utils/_ext_solver.md).

We provide a reference environment configuration as follows:

~~~python
python 3.7.15
pytorch 1.13.1
cuda 11.6
~~~

### Quick Start

Here are some useful demos for you.

+ If you want to train GhostingNet on your dataset, please run:

  ~~~python
  python train_recurrent.py
  ~~~

  and make sure that the dataset and dataloader methods are implemented by yourself.

+ If you want to infer the result of GhostingNet on a pretrained model, please run:

  ~~~python
  python infer_hh.py
  ~~~

  and you need to modify `--model_path` to your own model path.

+ And if you want to evaluate IoU, MAE, or any other relevant metrics on the results acquired from GhostingNet, please run:

  ~~~python
  cd evaluation_hh
  python single_eval.py
  ~~~

### Dataset and model

| *Content*                   | *Links*                                                      |
| --------------------------- | ------------------------------------------------------------ |
| Dataset (GEGD2)             | [gdrive](https://drive.google.com/file/d/1WDkK7DCZNuZKBMPT1lfEhySIUa0fYxW-/view?usp=sharing)\|[Baidu Disk](https://pan.baidu.com/s/1T9AsdsY_J_yhSQITb36q3w?pwd=1234) |
| Model (pretrained on GEGD2) | [gdrive](https://drive.google.com/file/d/13Mkq5nsAN1idaFPGnZVk-3CyiQPgGGRH/view?usp=sharing)\|[Baidu Disk](https://pan.baidu.com/s/1hMq_77qnp2c8FOkbUeKP_A?pwd=1234) |

### Contact

If you have any questions, please contact yan.tao@outlook.com
