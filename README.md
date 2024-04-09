%## GhostingNet: a Novel Approach for Glass Surface Detection with Ghosting Cues

%The official pytorch implementation of the paper **[GhostingNet: a Novel Approach for Glass Surface Detection with Ghosting Cues]()**

%#### Tao Yan, Jiahui Gao, Ke Xu, Helong Li, Xiangjie Zhu, Hao Huang, Benjamin Wah, Rynson W.H. Lau



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
