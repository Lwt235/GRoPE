# Joint Semantic and Rendering Enhancements in 3D Gaussian Modeling with Anisotropic Local Encoding
[![arXiv](https://img.shields.io/badge/arXiv-2601.02339-b31b1b)](http://arxiv.org/abs/2601.02339)

[[paper](http://arxiv.org/abs/2601.02339)]
[[code](https://github.com/YourRepoName/JSRE)]

**Jingming He**, **Chongyi Li**, **Shiqi Wang**, **Sam Kwong***  
(* denotes corresponding author)

---

This is the official repository for  
**“Joint Semantic and Rendering Enhancements in 3D Gaussian Modeling with Anisotropic Local Encoding”**  
(Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV 2025). We will release our code after completing the ongoing extensions and the submission of our follow-up work. (expected around July 2026)


## Overview

<div align="center">
  <img src="./imgs/fig1.jpg" alt="overview" width="960" />
</div>

Recent works propose extending 3DGS with semantic feature vectors for simultaneous semantic segmentation and image rendering. However, these methods often treat the semantic and rendering branches separately, relying solely on 2D supervision while ignoring the 3D Gaussian geometry. Moreover, current adaptive strategies adapt the Gaussian set depending solely on rendering gradients, which can be insufficient in subtle or textureless regions. In this work, we propose a joint enhancement framework for 3D semantic Gaussian modeling that synergizes both semantic and rendering branches. Firstly, unlike conventional point cloud shape encoding, we introduce an anisotropic 3D Gaussian Chebyshev descriptor using the Laplace–Beltrami operator to capture fine-grained 3D shape details, thereby distinguishing objects with similar appearances and reducing reliance on potentially noisy 2D guidance. In addition, without relying solely on rendering gradient, we adaptively adjust Gaussian allocation and spherical harmonics (SH) with local semantic and shape signals, enhancing rendering efficiency through selective resource allocation. Finally, we employ a cross-scene knowledge transfer module to continuously update learned shape patterns, enabling faster convergence and robust representations without relearning shape information from scratch for each new scene. Experiments on multiple datasets demonstrate improvements in segmentation accuracy and rendering quality while maintaining high rendering frame rates.


## Citation

If you find this work useful in your research, please consider citing:

```
@inproceedings{he2025joint,
  title={Joint Semantic and Rendering Enhancements in 3D Gaussian Modeling with Anisotropic Local Encoding},
  author={He, Jingming and Li, Chongyi and Wang, Shiqi and Kwong, Sam},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={28354--28363},
  year={2025}
}
```
