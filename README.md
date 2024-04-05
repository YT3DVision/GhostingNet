## GhostingNet: a Novel Approach for Glass Surface Detection with Ghosting Cues

The official pytorch implementation of the paper **[GhostingNet: a Novel Approach for Glass Surface Detection with Ghosting Cues (TPAMI2024)]()**

#### Tao Yan, Jiahui Gao, Ke Xu, Helong Li, Xiangjie Zhu, Hao Huang, Benjamin Wah, Rynson W.H. Lau

> Ghosting effects typically appear on glass surfaces, as each piece of glass has two contact surfaces causing two slightly offset layers of reflections. In this paper, we propose to take advantage of this intrinsic property of glass surfaces and apply it to glass surface detection, with two main technical novelties. First, we formulate a ghosting image formation model to describe the intensity and spatial relations among the main reflections and the background transmission within the glass region. Based on this model, we construct a new Glass Surface Ghosting Dataset (GSGD) to facilitate glass surface detection, with *∼*3.7*K* glass images and corresponding ghosting masks and glass surface masks. Second, we propose a novel method, called GhostingNet, for glass surface detection. Our method consists of a Ghosting Effects Detection (GED) module and a Ghosting Surface Detection (GSD) module. The key component of our GED module is a novel Double Reflection Estimation (DRE) block that models the spatial offsets of reflection layers for ghosting effect detection. The detected ghosting effects are then used to guide the GSD module for glass surface detection. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods. 

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

### Results Comparison

<p align="center">
  <img src="https://github.com/YT3DVision/GhostingNet/blob/main/utils/compare_results.jpg" style="width:100%" />
</p>

### Installation

Deformable Convolution Network(DCN) is required in this paper, if you want to run the project on Windows platform please refer to [this](./utils/_ext_solver.md).

### Code

<font color='red'>***Our code is comming soon!***</font>

### Contact

If you have any questions, please contact yan.tao@outlook.com
