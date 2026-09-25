# Quantum Neural Network (QNN) Explorations

An early exploration of hybrid quantum-classical neural networks: the idea, a planned project layout, and a first script that runs a small quantum circuit on a simulator next to a classical Keras network.

> **Status: concept plus minimal prototype.** Everything runs on a classical computer. The quantum part is a 3-qubit circuit executed on Qiskit's local simulator, and it is not yet connected to the neural network. Most files in the planned package structure are empty placeholders. There is no quantum speedup, and no claim of one, in this code.

[![QNN concept art](./ChatGPT%20Image%20Apr%2026%2C%202025%2C%2010_40_08%20AM.png)](./ChatGPT%20Image%20Apr%2026%2C%202025%2C%2010_40_08%20AM.png)

## The idea

Quantum machine learning replaces some layers of a neural network with parameterized quantum circuits, whose outputs are probability distributions over measured states. This project wants to explore that design space:

- **Probabilistic outputs:** classifications as distributions rather than hard labels.
- **Superposition and entanglement** as ingredients for richer interactions between "neurons".
- **Adaptive architectures** that reconfigure in response to data.

Whether quantum layers give a practical learning advantage over classical networks is an open research question, and on today's hardware and simulators they generally do not. The goal here is to learn by building, not to claim a breakthrough.

## What runs today

| File | What it does |
| --- | --- |
| `QNN.py` | Builds a 3-qubit GHZ circuit (Hadamard plus two CNOTs), runs it 1024 times on Qiskit Aer's `qasm_simulator`, prints the counts (about half `000`, half `111`), then trains a small Keras dense network for 5 epochs on random data and prints predictions |
| `Quantum Neural Network API.py` | FastAPI app with `GET /quantum_inference` (runs the same circuit and returns counts) and `POST /classical_prediction` (the Keras model on a list of 3 numbers). The model is trained on random data at import time |
| `training/loss_functions.py` | MSE, cross-entropy, KL-divergence wrappers around PyTorch and a `quantum_fidelity_loss` stub. Has a syntax error from a broken docstring, so it does not import yet |
| `Reducing Hallucinations in LLMs by Reducing Entropy.pdf` | A separate 3-page concept note proposing "external entropy dumps" to reduce LLM hallucination |

Everything else (`core/`, `qnn/`, `utils/`, `evaluation/`, `scripts/`, `examples/`, `tests/`, `notebooks/`, `setup.py`) is an empty placeholder for the planned structure.

## Quick start

The scripts use the Qiskit 0.x API (`from qiskit import Aer, execute`), which was removed in Qiskit 1.0, so pin older versions. Use Python 3.10 to 3.12.

```bash
git clone https://github.com/Mattbusel/Quantum-Neural-Network-Model-The-Future-of-AI-Cognition.git qnn
cd qnn
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install "qiskit<1.0" "qiskit-aer<0.14" tensorflow matplotlib

python QNN.py
```

Expected quantum output (exact counts vary):

```
Quantum Circuit Results: {'000': 514, '111': 510}
```

To run the API, copy it to an importable file name first:

```bash
pip install fastapi uvicorn
cp "Quantum Neural Network API.py" qnn_api.py
uvicorn qnn_api:app --reload
curl http://127.0.0.1:8000/quantum_inference
```

## Planned structure

```
core/          QNN layers and quantum backend wrapper
training/      trainer and loss functions
utils/         config, data loading, plotting
evaluation/    metrics
scripts/       train / evaluate CLIs
examples/      standalone demos
notebooks/     experiments
tests/         unit tests
```

## Next steps

- Replace the standalone circuit with a parameterized quantum layer whose parameters are trained together with the classical network (for example via Qiskit Machine Learning or PennyLane)
- Benchmark against an equally sized classical model on a small, real dataset
- Fill in the package structure and tests

## Contributing

Quantum computing people, ML engineers and anyone curious are welcome. Good first contributions: fix `training/loss_functions.py`, port `QNN.py` to the Qiskit 1.x API, or implement `core/qnn_layer.py`.
