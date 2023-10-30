import time

import numpy as np
from pyqula import geometry
from pyqula.transporttk.localprobe import LocalProbe

g = geometry.chain().get_supercell(2)  # create a 1D chain
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_onsite(1.8)
h.add_rashba(0.6)
h.add_exchange([0., 0., 0.6])
h.add_swave(0.2)  # create an artificial TSC

delta = 1e-3
energies = np.linspace(-1.4, 1.4, 100)  # energies

start_time = time.process_time()
lp = LocalProbe(h)

for _ in range(10):
    [lp.didv(energy=e, T=0.1) for e in energies]  # compute the differential conductance

print(f"Time elapsed: {time.process_time() - start_time:.2f} seconds")
