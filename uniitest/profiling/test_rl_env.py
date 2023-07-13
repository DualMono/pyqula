from pyqula import geometry
import numpy as np
import timeit
import cProfile

enable_profiler = True

if enable_profiler:
    pr = cProfile.Profile()
    pr.disable()

g = geometry.triangular_lattice()  # get the geometry
h = g.get_hamiltonian()  # get the Hamiltonian
h.add_rashba(1)
h.add_swave(0.3)
h.add_zeeman([0, 0, 0.1])


def fun():
    h.get_dos(delta=1e-1, energies=np.linspace(-3.5, 3.5, 100), write=False, nk=50)
    h.add_zeeman([0, 0, 0.1])
    h.add_onsite(0.1)
    print(f"Chern: {h.get_chern():.2f} Gap {h.get_gap():.2f}")


# Call once for JIT
fun()

if enable_profiler:
    pr.enable()

total_time = timeit.timeit(fun,
                           number=10)
if enable_profiler:
    pr.disable()
    pr.dump_stats('profile.pstat')

print(f"{total_time=}")
