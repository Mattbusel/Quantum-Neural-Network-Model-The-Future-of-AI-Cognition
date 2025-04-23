from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
import tensorflow as tf
from fastapi.responses import JSONResponse


app = FastAPI()


def create_quantum_circuit():
    """Create a simple quantum circuit."""
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)  
    circuit.cx(0, 1)  
    circuit.cx(1, 2)  
    circuit.measure([0, 1, 2], [0, 1, 2])  
    return circuit

def run_quantum_circuit(circuit: QuantumCircuit):
    """Execute the quantum circuit on a simulator and return results."""
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, simulator, shots=1024).result()
    counts = result.get_counts(circuit)
    return counts


def classical_nn(input_shape):
    """Create a simple classical neural network for predictions."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


X_train = np.random.random((100, 3))
y_train = np.random.randint(0, 2, 100)
model = classical_nn(input_shape=(3,))
model.fit(X_train, y_train, epochs=5)


class QuantumRequest(BaseModel):
    
    pass

class ClassicalRequest(BaseModel):
    input_data: list  


@app.get("/quantum_inference")
def get_quantum_inference():
    """Run a quantum circuit and return probabilistic results."""
    circuit = create_quantum_circuit()
    result = run_quantum_circuit(circuit)
    return JSONResponse(content={"quantum_results": result})

@app.post("/classical_prediction")
def classical_prediction(request: ClassicalRequest):
    """Use a classical neural network to make predictions based on input data."""
    input_data = np.array(request.input_data).reshape(1, -1)  # Reshape input for prediction
    prediction = model.predict(input_data)
    return {"prediction": prediction[0][0]}

