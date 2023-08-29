# bayesian-flow-mnist

A simple <a href="https://arxiv.org/abs/2308.07037">Bayesian Flow</a> model for MNIST in Pytorch.

- [x] Binarised MNIST generation using Bayesian Flow Discrete Data Loss
- [ ] Continuous MNIST generation using Bayesian Flow Continuous Data Loss

## How to Run

### Environment Setup

Aside from `pytorch`, `matplotlib`, and `tqdm`, the training script requires
<a href="https://github.com/thorinf/bayesian-flow-pytorch">bayesian-flow-pytorch</a>.

```commandline
pip install git+https://github.com/thorinf/bayesian-flow-pytorch
```

### Training

The model can be trained with the following command and MNIST will download automatically:

```terminal
python train.py -ckpt CHECKPOINTING_PATH -d MNIST_DOWNLOAD_PATH
```



