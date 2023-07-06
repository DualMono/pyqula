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
h.get_dos(delta=1e-1, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=50)

# chern_time = timeit.timeit(lambda: h.get_chern(), number=100) / 100
# print(f"{chern_time=}")

dos_time = timeit.timeit(lambda:
                         h.get_dos(delta=1e-1, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=50),
                         number=10) / 10
print(f"{dos_time=}")
