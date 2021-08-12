from utils import sparse_data_generator_from_hdf5_spikes, compute_classification_accuracy
from network import SHDSpikingNetwork
import h5py
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from utils import plot_voltage_traces


test_file = h5py.File("hdspikes/shd_test.h5", 'r')

x_test = test_file['spikes']
y_test = test_file['labels']

nb_inputs = 700
nb_steps = 100
max_time = 1.4
batch_size = 256

network = SHDSpikingNetwork(
    nb_inputs=nb_inputs, nb_hidden=200, nb_outputs=20, batch_size=batch_size)
network.load_state_dict(torch.load("shdnetwork.pth"))

print("Accuracy:", compute_classification_accuracy(
    network, x_test, y_test, max_time, nb_steps))


# get a single batch
for x_batch, y_batch in sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, batch_size, nb_steps, nb_inputs, max_time, shuffle=False):
    break
output, (mem_rec, spk_rec) = network(x_batch.to_dense())

# plot voltage traces
fig = plt.figure(dpi=100)
plot_voltage_traces(output)
plt.show()

# plot the hidden layer spiking activity for some input stimuli
nb_plt = 4
gs = GridSpec(1, nb_plt)
fig = plt.figure(figsize=(7, 3), dpi=150)
for i in range(nb_plt):
    plt.subplot(gs[i])
    plt.imshow(spk_rec[i].detach().cpu().numpy().T, cmap=plt.cm.gray_r, origin="lower")
    if i == 0:
        plt.xlabel("Time")
        plt.ylabel("Units")

    sns.despine()
plt.show()
