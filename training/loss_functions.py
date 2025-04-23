import torch
import torch.nn.functional as F



def mse_loss(output, target):
    
    return F.mse_loss(output, target)


def cross_entropy_loss(output, target):
   
    return F.cross_entropy(output, target)


def kl_divergence_loss(output_log_probs, target_probs):
  or probabilistic output layers or QNNs with softmax projections.
    "
    return F.kl_div(output_log_probs, target_probs, reduction='batchmean')


# Placeholder for a future quantum-aware loss function
def quantum_fidelity_loss(state_1, state_2):
    """
    Quantum Fidelity Loss (Placeholder)
    Measures the similarity between two quantum states.
    Could be used when working with quantum simulators or quantum circuits
    that output state vectors or density matrices.
    """
    raise NotImplementedError("Quantum fidelity loss is under development.")


def get_loss_function(name):
    """
    Dynamically fetch a loss function by name.
    """
    losses = {
        "mse": mse_loss,
        "cross_entropy": cross_entropy_loss,
        "kl_div": kl_divergence_loss,
        "quantum_fidelity": quantum_fidelity_loss
    }
    if name not in losses:
        raise ValueError(f"Loss function '{name}' is not implemented.")
    return losses[name]

