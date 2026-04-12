import torch

def compute_l2_error(pred, exact):
    """
    Computa el error relativo norma L2 para arreglos multidimensionales de PyTorch.
    El resultado regresivo se expresa sin graidentes o tensores embebidos.
    """
    pred_vec = pred.flatten()
    exact_vec = exact.flatten()
    error = torch.linalg.norm(pred_vec - exact_vec, 2) / torch.linalg.norm(exact_vec, 2)
    return error.item()

def compute_residual_mean(residuals):
    """
    Calcula el Mean Squared Error MSE en el array de residuos.
    """
    mse = torch.mean(torch.square(residuals))
    return mse.item()
