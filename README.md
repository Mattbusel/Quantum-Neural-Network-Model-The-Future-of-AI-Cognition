# Quantum Neural Network (QNN) Explorations

Early quantum plus classical neural net prototype.

- Runs a 3-qubit circuit on Qiskit's simulator.
- Trains a small Keras network next to it.
- Serves both through a FastAPI app.

Status: concept plus small prototype. The two parts are not connected yet. No quantum speedup claimed.

## Install

```bash
git clone https://github.com/Mattbusel/Quantum-Neural-Network-Model-The-Future-of-AI-Cognition.git qnn
cd qnn
pip install "qiskit<1.0" "qiskit-aer<0.14" tensorflow matplotlib
```

## Use

```bash
python QNN.py
```

Prints counts like `{'000': 514, '111': 510}`.

More: [docs/DETAILS.md](docs/DETAILS.md)

MIT license, see [LICENSE](LICENSE).
