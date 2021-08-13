from utils import sparse_data_generator_from_hdf5_spikes
from network import SHDSpikingNetwork
import h5py
import torch
from sparsefool import sparsefool
from utils import get_prediction


test_file = h5py.File("hdspikes/shd_test.h5", 'r')

x_test = test_file['spikes']
y_test = test_file['labels']

nb_inputs = 700
nb_steps = 100
max_time = 1.4
batch_size = 1

network = SHDSpikingNetwork(
    nb_inputs=nb_inputs, nb_hidden=200, nb_outputs=20, batch_size=batch_size)
network.load_state_dict(torch.load("shdnetwork.pth"))


accs = []
for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
        x_test, y_test, batch_size, nb_steps, nb_inputs, max_time, shuffle=False):

    x_0 = x_local.to_dense()

    am = get_prediction(network, x_0)

    if am == y_local:
        results = sparsefool(
            x_0=x_0,
            net=network,
            max_hamming_distance=1000,
            lb=0.0,
            ub=x_0.max(),
            lambda_=1.0,
            max_iter=20,
            epsilon=0.02,
            overshoot=0.02,
            step_size=0.001,
            max_iter_deep_fool=50,
            device="cuda",
            verbose=False,
        )
        x_adv = results["X_adv"]
        print("N. spikes before attack:", x_0.sum())
        print("Max spikes in bin before attack:", x_0.max())
        print("N. spikes after attack:", x_adv.sum())
        print("Max spikes in bin after attack:", x_adv.max())

        if results["success"]:
            am_att = get_prediction(network, x_adv)
            if am_att != am:
                print("Success.")
            else:
                print("WARNING Success but same label.")

    else:
        print("Wrong prediction, not attacking.")
