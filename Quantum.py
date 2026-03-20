import numpy as np
from qiskit import QuantumCircuit


def quantum_encoder(url):
    encoded_state = np.random.rand(2)
    norm = np.linalg.norm(encoded_state)

    if norm == 0:
        raise ValueError("Encoded state cannot be a zero vector.")

    encoded_state /= norm
    return encoded_state


def QITLayer(phi, weights):
    num_qubits = 1 if len(phi) == 2 else int(np.log2(len(phi)))
    qc = QuantumCircuit(num_qubits)

    print(f"Initializing QITLayer with phi: {phi}, num_qubits: {num_qubits}")

    qc.initialize(phi, [0])

    for i in range(num_qubits):
        qc.rx(weights[i], i)

    return qc


def QNNLayer(phi, weights, bias):
    num_qubits = 1 if len(phi) == 2 else int(np.log2(len(phi)))
    qc = QuantumCircuit(num_qubits)

    print(f"Initializing QNNLayer with phi: {phi}, num_qubits: {num_qubits}")

    qc.initialize(phi, [0])

    for i in range(num_qubits):
        qc.ry(weights[i], i)

    qc.measure_all()
    return qc


def quantum_loss_function(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def quantum_classifier(circuit, shots=1024):
    counts = {'0': 0, '1': 0}

    for _ in range(shots):
        if np.random.rand() < 0.5:
            counts['0'] += 1
        else:
            counts['1'] += 1

    prob_phishing = counts['1'] / shots
    return prob_phishing


def train_quantum_model(urls, epochs=10, learning_rate=0.01):
    num_layers = 2
    num_qubits = 1

    weights_QIT = [np.random.rand(num_qubits) for _ in range(num_layers)]
    weights_QNN = np.random.rand(num_qubits)
    bias = np.random.rand(1)

    for epoch in range(epochs):
        for url in urls:
            phi = quantum_encoder(url)

            for k in range(num_layers):
                QITLayer(phi, weights_QIT[k])

            l = QNNLayer(phi, weights_QNN, bias)

            target = np.array([1])
            prediction = quantum_classifier(l)

            loss = quantum_loss_function(np.array([prediction]), target)

            for k in range(num_layers):
                weights_QIT[k] -= learning_rate * np.random.rand(num_qubits)

            weights_QNN -= learning_rate * np.random.rand(num_qubits)
            bias -= learning_rate * np.random.rand(1)

    return weights_QIT, weights_QNN, bias


def classify_url(url, threshold=0.5):
    urls = [url]

    weights_QIT, weights_QNN, bias = train_quantum_model(urls)

    phi = quantum_encoder(url)
    l = QNNLayer(phi, weights_QNN, bias)

    P_i = quantum_classifier(l)

    return "Malicious" if P_i > threshold else "Benign"


if __name__ == "__main__":
    url = "http://ongelezen-voda.000webhostapp.com/inloggen.html"
    result = classify_url(url)

    print(f"The URL '{url}' is classified as: {result}")