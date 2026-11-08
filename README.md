# Quantum Neural Network (QNN) Explorations

An early try at mixed quantum and classical neural networks. The idea, a planned layout, and a first script that runs a small quantum circuit on a simulator next to a Keras network.

> **Status: concept plus small prototype.** Everything runs on a normal computer. The quantum part is a 3-qubit circuit on Qiskit's local simulator. It is not connected to the neural network yet. Most files in the planned layout are empty. No quantum speedup here, and no claim of one.

[![QNN concept art](./ChatGPT%20Image%20Apr%2026%2C%202025%2C%2010_40_08%20AM.png)](./ChatGPT%20Image%20Apr%2026%2C%202025%2C%2010_40_08%20AM.png)

## The idea

Quantum machine learning swaps some layers of a neural network for quantum circuits with tunable settings. Their outputs are probabilities over measured states. Things to try:

- **Probability outputs:** answers as distributions, not hard labels.
- **Superposition and entanglement** to link "neurons" in richer ways.
- **Networks that rewire** based on the data.

Whether quantum layers learn better than classical ones in practice is an open question. On today's hardware and simulators they mostly don't. The goal is to learn by building, not to claim a breakthrough.

## What runs today

| File | What it does |
| --- | --- |
| `QNN.py` | Builds a 3-qubit GHZ circuit (Hadamard plus two CNOTs), runs it 1024 times on Qiskit Aer's `qasm_simulator`, prints the counts (about half `000`, half `111`), then trains a small Keras dense network for 5 epochs on random data and prints predictions |
| `Quantum Neural Network API.py` | FastAPI app with `GET /quantum_inference` (runs the same circuit and returns counts) and `POST /classical_prediction` (the Keras model on a list of 3 numbers). The model is trained on random data at import time |
| `training/loss_functions.py` | MSE, cross-entropy, KL-divergence wrappers around PyTorch and a `quantum_fidelity_loss` stub. A broken docstring causes a syntax error, so it does not import yet |
| `Reducing Hallucinations in LLMs by Reducing Entropy.pdf` | A separate 3-page note proposing "external entropy dumps" to cut LLM hallucination |

Everything else (`core/`, `qnn/`, `utils/`, `evaluation/`, `scripts/`, `examples/`, `tests/`, `notebooks/`, `setup.py`) is an empty placeholder.

## Quick start

The scripts use the Qiskit 0.x API (`from qiskit import Aer, execute`). Qiskit 1.0 removed it, so pin older versions. Use Python 3.10 to 3.12.

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

To run the API, copy it to a file name Python can import:

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

- Swap the standalone circuit for a quantum layer trained together with the classical network (for example with Qiskit Machine Learning or PennyLane)
- Compare against a same-size classical model on a small, real dataset
- Fill in the package layout and tests

## Contributing

Quantum people, ML engineers and anyone curious welcome. Good first tasks: fix `training/loss_functions.py`, port `QNN.py` to the Qiskit 1.x API, or write `core/qnn_layer.py`.
