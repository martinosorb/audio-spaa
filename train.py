#!/usr/bin/env python
# coding: utf-8

# # Tutorial 4: Training a spiking neural network on a spiking dataset (Spiking Heidelberg Digits)
#
# Manu Srinath Halvagal (https://zenkelab.org/team/) and Friedemann Zenke (https://fzenke.net)

# For more details on surrogate gradient learning, please see:
#
# > Neftci, E.O., Mostafa, H., and Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-based optimization to spiking neural networks. IEEE Signal Process Mag 36, 51–63.
# > https://ieeexplore.ieee.org/document/8891809 and https://arxiv.org/abs/1901.09948
#
# > Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2020). The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks. IEEE Transactions on Neural Networks and Learning Systems 1–14.
# > https://ieeexplore.ieee.org/document/9311226 and https://arxiv.org/abs/1910.07407"

# In Tutorials 2 and 3, we have seen how to train a multi-layer spiking neural network on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) along with an activity regularizer. However, the spiking activity input to the network was generated using a simple time-to-first-spike code. Here we apply the model to train a spiking network to learn the Spiking Heidelberg Digits dataset (https://compneuro.net/posts/2019-spiking-heidelberg-digits/). This dataset uses a more sophisticated cochlear model to generate the spike data corresponding to audio recordings of spoken digits (examples below).

# ![Image of Yaktocat](https://compneuro.net/img/hdspikes_digits_samples.png)

import os
import h5py

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import seaborn as sns

import torch
import torch.nn as nn

# from data_retrieval_utils import get_shd_dataset
from utils import compute_classification_accuracy, sparse_data_generator_from_hdf5_spikes
from network import SHDSpikingNetwork

# The coarse network structure and the time steps are dicated by the SHD dataset.
nb_inputs = 700
nb_steps = 100
max_time = 1.4
batch_size = 256

network = SHDSpikingNetwork(
    nb_inputs=nb_inputs, nb_hidden=200, nb_outputs=20, batch_size=batch_size)
network.load_state_dict(torch.load("shdnetwork.pth"))

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ### Setup of the spiking dataset
# Here we load the Dataset
cache_dir = os.path.expanduser(".")
cache_subdir = "hdspikes"
# get_shd_dataset(cache_dir, cache_subdir)

train_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r')
test_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')

x_train = train_file['spikes']
y_train = train_file['labels']
x_test = test_file['spikes']
y_test = test_file['labels']


# The code for learning the SHD dataset is nearly identical to what we have seen
# for the FashionMNIST dataset in the last two tutorials. An important difference
# is that, now, we have the input data already in the form of spikes.
# This is reflected in the sparse_data_generator below.
#
# In order to use the data for learning with our spiking network, we also need to
# discretize the spike times into n_steps bins. Note the additional max_time argument
# (~1.4 for SHD) that forms the upper limit of the bins.


# ## Training the network
nb_epochs = 100
lr = 1e-3

params = network.parameters()
optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

loss_hist = []
for e in range(nb_epochs):
    local_loss = []
    for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
            x_train, y_train, batch_size, nb_steps, nb_inputs, max_time):
        output, recs = network(x_local.to_dense())
        _, spks = recs
        m, _ = torch.max(output, 1)
        log_p_y = log_softmax_fn(m)

        # Here we set up our regularizer loss
        # The strength paramters here are merely a guess and there should
        # be ample room for improvement by tuning these paramters.
        reg_loss = 2e-6*torch.sum(spks)  # L1 loss on total number of spikes
        reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0)**2)  # L2 loss on spikes per neuron

        # Here we combine supervised loss and the regularizer
        loss_val = loss_fn(log_p_y, y_local) + reg_loss

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        local_loss.append(loss_val.item())
    mean_loss = np.mean(local_loss)
    loss_hist.append(mean_loss)
    torch.save(network.state_dict(), "shdnetwork.pth")
    print("Epoch %i: loss=%.5f" % (e+1, mean_loss))

    if (e+1) % 10 == 0:
        print("Training accuracy: %.3f" % (compute_classification_accuracy(
            network, x_train, y_train, max_time)))
        print("Test accuracy: %.3f" % (compute_classification_accuracy(
            network, x_test, y_test, max_time)))

#
# def get_mini_batch(x_data, y_data, shuffle=False):
#     for ret in sparse_data_generator_from_hdf5_spikes(x_data, y_data, batch_size, nb_steps, nb_inputs, max_time, shuffle=shuffle):
#         return ret
#
#
# x_batch, y_batch = get_mini_batch(x_test, y_test)
# output, other_recordings = run_snn(x_batch.to_dense())
# mem_rec, spk_rec = other_recordings
#
#
# fig = plt.figure(dpi=100)
# plot_voltage_traces(output)
#
# # Let's plot the hiddden layer spiking activity for some input stimuli
#
# nb_plt = 4
# gs = GridSpec(1,nb_plt)
# fig= plt.figure(figsize=(7,3),dpi=150)
# for i in range(nb_plt):
#     plt.subplot(gs[i])
#     plt.imshow(spk_rec[i].detach().cpu().numpy().T,cmap=plt.cm.gray_r, origin="lower" )
#     if i==0:
#         plt.xlabel("Time")
#         plt.ylabel("Units")
#
#     sns.despine()
#
# plt.show()
# # We see that spiking in the hidden layer is quite sparse as in the previous Tutorial 3 because we used the same activity regularizer.
#
# # <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
