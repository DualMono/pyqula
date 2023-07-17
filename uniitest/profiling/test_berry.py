from pyqula import geometry
from pyqula.topology import berry_phase
import numpy as np
import timeit
import cProfile

enable_profiler = True

if enable_profiler:
    pr = cProfile.Profile()
    pr.disable()

g = geometry.chain()  # create a chain
N = 2  # number of sites per unit cell
g = g.get_supercell(N)  # create supercell

h = g.get_hamiltonian()  # get the Hamiltonian

# add onsite modulation
h.add_onsite(lambda r: 0.2 * np.cos(2 * np.pi / N * (r[0] - g.r[0][0])))

h.add_onsite(1.8)  # add chemical potential

h.add_rashba(0.3)  # add Rashba SOC
h.add_zeeman([0, 0, 0.3])  # add Zeeman field
h.add_swave(0.1)  # add superconducting pairing


def fun():
    berry_phase(h, write=False)
    # print(f"Berry phase {berry_phase(h, write=False) / np.pi:.1f} π")


# Call once for JIT
fun()

if enable_profiler:
    pr.enable()

total_time = timeit.timeit(fun,
                           number=1000)
if enable_profiler:
    pr.disable()
    pr.dump_stats('profile.pstat')

print(f"{total_time=}")
