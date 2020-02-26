# Quantum machine learning concepts

In October 2019,
<a href="https://www.blog.google/perspectives/sundar-pichai/what-our-quantum-computing-milestone-means/" class="external">Google announced</a>
they achieved
<a href="https://www.nature.com/articles/s41586-019-1666-5" class="external">quantum supremacy</a>.
Using 53&nbsp;*noisy*&nbsp;qubits, this demonstration was a critical first step to unlock
the full potential of quantum computing and marks the beginning of the
<a href="https://quantum-journal.org/papers/q-2018-08-06-79/" class="external">Noisy Intermediate-Scale Quantum</a>&nbsp;(NISQ)
computing era. In the coming years, quantum devices with tens-to-hundreds of
noisy qubits are expected to become a reality. So what is possible with these
devices?

There are many ideas for leveraging NISQ quantum computing including
optimization, quantum simulation, cryptography, and machine learning.
TensorFlow&nbsp;Quantum&nbsp;(TFQ) is designed to help researchers experiment
with these ideas. Researchers create and run *quantum circuits*. It integrates
with TensorFlow, an established machine learning framework used for research and
production. TensorFlow Quantum provides flexible and performant tools and
constructs used by quantum machine learning researchers. TensorFlow Quantum
hopes to bridge the quantum and classical machine learning communities—and
enrich both with new perspectives and ideas.

## NISQ quantum machine learning

During the NISQ-era, quantum algorithms with known speedups over classical
algorithms—like
<a href="https://arxiv.org/abs/quant-ph/9508027" class="external">Shor's factoring algorithm</a> or
<a href="https://arxiv.org/abs/quant-ph/9605043" class="external">Grover's search algorithm</a>—are
not yet possible at a meaningful scale.

A goal of TensorFlow Quantum is to help discover algorithms for the
NISQ-era, with particular interest in:

1. *Use classical machine learning to enhance NISQ algorithms.* The hope is that
   techniques from classical machine learning can enhance our understanding of
   quantum computing. For example,
   <a href="https://arxiv.org/abs/1907.05415" class="external">this paper</a>
   shows a recurrent neural network (RNN) used to discover that optimization of
   the control parameters for algorithms like the QAOA and VQE are more efficient
   than simple off the shelf optimizers. And
   <a href="https://www.nature.com/articles/s41534-019-0141-3" class="external">this paper</a>
   uses reinforcement learning to help mitigate errors and produce higher
   quality quantum gates.
2. *Model quantum data with quantum circuits.* Classically modeling quantum data
   is possible if you have an exact description of the datasource—but sometimes
   this isn’t possible. To solve this problem, you can try modeling on the
   quantum computer itself and measure/observe the important statistics.
   <a href="https://www.nature.com/articles/s41567-019-0648-8" class="external">This paper</a>
   shows a quantum circuit designed with a structure analogous to a
   convolutional neural network (CNN) to detect different topological phases of
   matter. The quantum computer holds the data and the model. The classical
   processor sees only measurement samples from the model output and never the
   data itself. In
   <a href="https://arxiv.org/pdf/1711.07500.pdf" class="external">this paper</a>
   the authors learn to compress information about quantum many-body systems
   using a DMERA model.

Other areas of interest in quantum machine learning include:

1. Modeling purely classical data on quantum computers.
2. Quantum-inspired classical algorithms. TFQ does not contain any purely
   classical algorithms that are quantum-inspired.

While these last two areas did not inform the design of TensorFlow Quantum,
you can still use TFQ for research here. For example, in
<a href="https://arxiv.org/abs/1802.06002" class="external">this paper</a>
the authors use a quantum computer to solve some purely classical data problems—
which could be implemented in TFQ.
