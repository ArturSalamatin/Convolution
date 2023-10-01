# Convolution

The repo demonstrates the template metaprogramming skill (C++)
for the design of numerical algorithm of convolution calculation.

The Kernel and Operand (i.e., the Flux) have different nature,
sizes, and ways of allocations depending on the simulation regime
(ConstStep, MainStep, MixStep, SmallStep).

All regimes are collectively described as an hierarchy of templated calsses.
In this way the DRY principle is obeyed, and the overheads associated 
with virtual methods are eliminated.

Convolution folder contains the executable code.
Tests folder demonstrates the usage of Convolution-classes.
Eigen folder contains the Eigen-library code
https://eigen.tuxfamily.org/index.php?title=Main_Page
