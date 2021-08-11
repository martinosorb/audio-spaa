from surr_grad import spike_fn
import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
nb_steps = 100
time_step = 1e-3


class SHDSpikingNetwork(nn.Module):
    def __init__(self, nb_hidden, nb_inputs, nb_outputs,
                 tau_mem=10e-3, tau_syn=5e-3, batch_size=256):
        super().__init__()
        self.nb_hidden = nb_hidden
        self.nb_inputs = nb_inputs
        self.nb_outputs = nb_outputs
        self.alpha = float(np.exp(-time_step/tau_syn))
        self.beta = float(np.exp(-time_step/tau_mem))
        self.batch_size = batch_size

        # Let's also now include recurrent weights in the hidden layer.
        # This significantly improves performance on the SHD dataset .

        weight_scale = 0.2

        w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

        w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

        v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(v1, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.v1 = nn.Parameter(v1)

    # run_snn is also changed now in order to include the recurrent input
    # in the hidden layer computation.
    def forward(self, inputs):
        syn = torch.zeros((self.batch_size, self.nb_hidden), device=device, dtype=dtype)
        mem = torch.zeros((self.batch_size, self.nb_hidden), device=device, dtype=dtype)

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        out = torch.zeros((self.batch_size, self.nb_hidden), device=device, dtype=dtype)
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))
        for t in range(nb_steps):
            h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, self.v1))
            mthr = mem-1.0
            out = spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = self.alpha*syn + h1
            new_mem = (self.beta * mem + syn)*(1.0-rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        flt = torch.zeros((self.batch_size, self.nb_outputs), device=device, dtype=dtype)
        out = torch.zeros((self.batch_size, self.nb_outputs), device=device, dtype=dtype)
        out_rec = [out]
        for t in range(nb_steps):
            new_flt = self.alpha*flt + h2[:, t]
            new_out = self.beta*out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs
