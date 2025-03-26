## HQSL: A Hybrid Quantum Neural Network for Split Learning

This is an example (full code can be provided upon reasonable requests) of how we can combine Hybrid Quantum Neural Networks with Split Learning (HQSL)

HQSL allows scaling of multiple resource-constrained clients to train their ML models with a Hybrid Quantum Server

centralized_hybrid.py consists of HQSL prior to splitting it before the quantum layer

split_hybrid.py consists of HQSL modelling and training. 

split_N_hybrid.py consists of HQSL scaled for N clients.

The baselines are the classical equivalents of the above.
