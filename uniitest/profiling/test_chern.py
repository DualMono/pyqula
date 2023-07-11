from pyqula import geometry
import numpy as np
import timeit
import cProfile

enable_profiler = False

if enable_profiler:
    pr = cProfile.Profile()
    pr.disable()

g = geometry.triangular_lattice()  # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian

h.add_rashba(1)  # Rashba spin-orbit coupling
h.add_zeeman([0, 0, 1])  # Zeeman field
h.add_onsite(1)

# Call the function once before benchmarking for JIT
h.get_chern()

if enable_profiler:
    pr.enable()

chern_time = timeit.timeit(lambda:
                           h.get_chern(),
                           number=100)
if enable_profiler:
    pr.disable()
    pr.dump_stats('profile.pstat')

print(f"{chern_time=}")
