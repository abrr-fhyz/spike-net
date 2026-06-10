# Spiking Neural Network (SNN)
A custom-designed, vectorized Leaky Integrate-and-Fire (LIF) Spiking Neural Network architecture integrated with a Reinforcement Learning reward system. The goal is to implement the complete nervous system of the *C. elegans* organism.

## Features
- **LIF SNN** with custom `Neuron` and `Synapse` classes.
- Vectorized operations for easier calculation.
- Reinforcement Learning based reward mechanism.
- Architecture assumes neurons to be in a Three-layer formation: Input, Hidden, and Output neuron layers.
- Seven synaptic configurations: Input→Hidden, Hidden→Output, Input→Input, Hidden→Hidden, Output→Output, Output→Hidden, Hidden→Input 
- STDP **(Spike-Timing-Dependent Plasticity)** learning rule.

## Project Structure
- **`NeuralNet`**: Core module containing the SNN implementation, neurons, and synapses.
- **`Data`**: Contains the *C. elegans* connectome data and parser.
- **`Env`**: Contains the environment module for simulating worm chemotaxis.

# Neural Network Model of the *C. elegans* brain
***C. elegans* connectome**: Models 302 neurons and 5,106 biological synapses from the `Data` module into an instance of `NeuralNet.SNN`.
