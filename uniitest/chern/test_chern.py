from pyqula import geometry
import numpy as np
import timeit

# parallelization
# from pyqula import parallel
# parallel.set_cores("max")  # uncomment to use all the cores

g = geometry.triangular_lattice()  # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian

h.add_rashba(1)  # Rashba spin-orbit coupling
h.add_zeeman([0, 0, 1])  # Zeeman field
h.add_onsite(1)

# Call the function once before benchmarking for JIT
h.get_chern()

chern_time = timeit.timeit(lambda:
                           h.get_chern(),
                           number=100) / 100
print(f"{chern_time=}")
