# Audio-SPAA

Code for training a network on the Heidelberg spoken digits spiking database, and attacking the data
with a sparse adversarial attack that fools the network.

## Code sources
The majority of the training code, with modifications, is by Manu Srinath Halvagal (https://zenkelab.org/team/)
and Friedemann Zenke (https://fzenke.net), who released it under the Creative Commons Attribution
4.0 International License. See [this file](https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial4.ipynb).

The sparsefool code, with some modifications, comes from the LTS4 lab at EPFL, who
released it under the Apache 2.0 license. See [this repo](https://github.com/LTS4/SparseFool).


## References
 For more details on surrogate gradient learning, please see:
 - Neftci, E.O., Mostafa, H., and Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient based optimization to spiking neural networks. IEEE Signal Process Mag 36, 51–63. https://ieeexplore.ieee.org/document/8891809 and https://arxiv.org/abs/1901.09948

For more details on SparseFool see:
 - Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard (2019). SparseFool: a few pixels make a big difference. CVPR 2019. https://arxiv.org/abs/1811.02248.

For more details on the Heidelberg dataset see:
 - Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2020). The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks. IEEE Transactions on Neural Networks and Learning Systems 1–14. https://ieeexplore.ieee.org/document/9311226
