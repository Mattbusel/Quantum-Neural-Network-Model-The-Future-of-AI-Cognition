import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.visualization import plot_histogram


def create_quantum_circuit():
   
    circuit = QuantumCircuit(3, 3)

   
    circuit.h(0) 
    circuit.cx(0, 1)  
    circuit.cx(1, 2) 

    circuit.measure([0, 1, 2], [0, 1, 2])

    return circuit

def run_quantum_circuit(circuit):
   
    simulator = Aer.get_backend('qasm_simulator')
    
    
    result = execute(circuit, simulator, shots=1024).result()

   
    counts = result.get_counts(circuit)
    return counts


import tensorflow as tf

def classical_nn(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification for simplicity
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
   
    quantum_circuit = create_quantum_circuit()

  
    quantum_results = run_quantum_circuit(quantum_circuit)
    print("Quantum Circuit Results:", quantum_results)

 
    model = classical_nn(input_shape=(3,)) 
    
  
    X_train = np.random.random((100, 3))
    y_train = np.random.randint(0, 2, 100)

    
    model.fit(X_train, y_train, epochs=5)

   
    predictions = model.predict(X_train)
    print("Classical NN Predictions:", predictions[:5])

   
    plot_histogram(quantum_results)


if __name__ == '__main__':
    main()
