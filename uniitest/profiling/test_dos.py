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
h.get_dos(delta=1e-1, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=50)

# chern_time = timeit.timeit(lambda: h.get_chern(), number=100) / 100
# print(f"{chern_time=}")

if enable_profiler:
    pr.enable()
dos_time = timeit.timeit(lambda:
                         h.get_dos(delta=1e-1, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=50),
                         number=10)
if enable_profiler:
    pr.disable()
    pr.dump_stats('profile.pstat')

print(f"{dos_time=}")
