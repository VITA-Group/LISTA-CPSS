# Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and Thresholds
This repository is for LISTA networks with weight coupling and/or support
selection structures introduced in the following paper:

[Xiaohan Chen\*](http://people.tamu.edu/~chernxh),
[Jialin Liu\*](https://www.math.ucla.edu/~liujl11/),
[Zhangyang Wang](http://www.atlaswang.com/) and
[Wotao Yin](http://www.math.ucla.edu/~wotaoyin/)
"Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and
Thresholds", accepted as spotlight oral at NIPS 2018. The preprint version is
[here](https://arxiv.org/abs/1808.10038).

\*: These authors contributed equally and are listed alphabetically.

The code is tested in Linux environment (Tensorflow: 1.10.0, CUDA9.0) with Titan
1080Ti GPU.

## Introduction
In recent years, unfolding iterative algorithms as neural networks has become an
empirical success in solving sparse recovery problems. However, its theoretical
understanding is still immature, which prevents us from fully utilizing the
power of neural networks. In this work, we study unfolded ISTA (Iterative
Shrinkage Thresholding Algorithm) for sparse signal recovery. We introduce a
weight structure that is necessary for asymptotic convergence to the true sparse
signal. With this structure, unfolded ISTA can attain a linear convergence,
which is better than the sublinear convergence of ISTA/FISTA in general cases.
Furthermore, we propose to incorporate thresholding in the network to perform
support selection, which is easy to implement and able to boost the convergence
rate both theoretically and empirically. Extensive simulations, including sparse
vector recovery and a compressive sensing experiment on real image data,
corroborate our theoretical results and demonstrate their practical usefulness.

## Citation
If you find our code helpful in your resarch or work, please cite our paper.
```
@article{chen2018theoretical,
  title={Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and Thresholds},
  author={Chen, Xiaohan and Liu, Jialin and Wang, Zhangyang and Yin, Wotao},
  journal={arXiv preprint arXiv:1808.10038},
  year={2018}
}
```
