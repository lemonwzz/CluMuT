## UNSUPERVISED LEARNING WITH CLUSTER-LEVEL CONSISTENCY AND DE-CORRELATION


PyTorch implementation of CluMuT.

```
.....
```

### Pretrained Model

<table>
  <tr>
    <th>dataset</th>
    <th>epochs</th>
    <th>batch size</th>
    <th>acc</th>
    <th colspan="4" >download</th>
  </tr>
  <tr>
    <td>CIFAR10</td>
    <td>300</td>
    <td>128</td>
    <td>87.22%</td>
    <td><a href=。。。>ResNet-18</a></td>
    <td><a href="。。。">full checkpoint</a></td>
    <td><a href="。。。">train logs</a></td>
    <td><a href="。。。"</a></td>
  </tr>
    <tr>
    <td>CIFAR100</td>
    <td>300</td>
    <td>128</td>
    <td>63.39%</td>
    <td><a href=。。。>ResNet-18</a></td>
    <td><a href="。。。">full checkpoint</a></td>
    <td><a href="。。。">train logs</a></td>
    <td><a href="。。。"</a></td>
  </tr>
</table>


You can choose to download either the weights of the pretrained ResNet-18 network or the full checkpoint, which also contains the weights of the projector network and the state of the optimizer. 

The pretrained model is also available on PyTorch Hub.

```
!python evaluate.py --arch resnet18 --data cifar10 --root /content/cifar10 --pretrained  ../checkpoints/cifar10/ --checkpoint-dir ../checkpoints/cifar10/val1 --lr-classifier 0.3
```

### CluMuT Training

Install PyTorch and download ImageNet by following the instructions in the [requirements](https://github.com/pytorch/examples/tree/master/imagenet#requirements) section of the PyTorch ImageNet training example. The code has been developed for PyTorch version 1.7.1 and torchvision version 0.8.2, but it should work with other versions just as well. 

Our best model is obtained by running the following command:

```
!python CluMuT.py --data cifar10 --arch resnet18 --root /content/cifar10 --learning-rate-weights 0.5 --weight-decay 1e-4 --batch-size 128 --epochs 300 --projector 1024-1024-1024 --checkpoint-dir ../twins_checkpoint/cifar10
```

Training time is approximately 2 days on v100 GPUs.

### Evaluation: Linear Classification

Train a linear probe on the representations learned by Barlow Twins. Freeze the weights of the resnet and use the entire ImageNet training set.

```
!python evaluate.py --arch resnet18 --data cifar10 --root /content/cifar10 --pretrained  ../checkpoints/cifar10/ --checkpoint-dir ../checkpoints/cifar10/val1 --lr-classifier 0.3
```
