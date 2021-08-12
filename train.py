#!/usr/bin/env python
# coding: utf-8
import h5py
import numpy as np
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
# network.load_state_dict(torch.load("shdnetwork.pth"))

train_file = h5py.File("hdspikes/shd_train.h5", 'r')
test_file = h5py.File("hdspikes/shd_test.h5", 'r')

x_train = train_file['spikes']
y_train = train_file['labels']
x_test = test_file['spikes']
y_test = test_file['labels']

# ## Training the network
nb_epochs = 100
lr = 1e-3
reg_param = 2e-6

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
        reg_loss = reg_param*torch.sum(spks)  # L1 loss on total number of spikes
        reg_loss += reg_param*torch.mean(  # L2 loss on spikes per neuron
            torch.sum(torch.sum(spks, dim=0), dim=0)**2)

        # Here we combine supervised loss and the regularizer
        loss_val = loss_fn(log_p_y, y_local) + reg_loss

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        local_loss.append(loss_val.item())
    mean_loss = np.mean(local_loss)
    loss_hist.append(mean_loss)
    torch.save(network.state_dict(), "shdnetwork_in_training.pth")
    print("Epoch %i: loss=%.5f" % (e+1, mean_loss))

    if (e+1) % 10 == 0:
        print("Training accuracy: %.3f" % (compute_classification_accuracy(
            network, x_train, y_train, max_time, nb_steps)))
        print("Test accuracy: %.3f" % (compute_classification_accuracy(
            network, x_test, y_test, max_time, nb_steps)))
