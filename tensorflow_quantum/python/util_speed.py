import cirq
from tensorflow_quantum.python import util
import time

big_circuit = cirq.testing.random_circuit(cirq.GridQubit.rect(1, 4), 3, 0.9)
t = time.time()
r = util.from_tensor(util.convert_to_tensor([[big_circuit, big_circuit]]))
t2 = time.time()
print(r.shape)
print(type(r[0]))
print(r[0])
print('Time for full conversion:', t2 - t)
