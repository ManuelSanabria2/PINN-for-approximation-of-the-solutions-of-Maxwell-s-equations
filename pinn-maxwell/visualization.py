import matplotlib.pyplot as plt
import numpy as np
import torch
from analytical import exact_solution, L

def plot_snapshot(model, t_snapshot, grid_pts=100, save_path=None):
    """
    Genera un snapshot 2D rellenado comparando la predicción vs Exacta para Ez en (x,y).
    """
    # Generar cuadricula
    x_lin = np.linspace(0, L, grid_pts)
    y_lin = np.linspace(0, L, grid_pts)
    X, Y = np.meshgrid(x_lin, y_lin)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    T_flat = np.full_like(X_flat, t_snapshot)
    
    # Formato a tensores
    x_t = torch.tensor(X_flat, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(Y_flat, dtype=torch.float32).unsqueeze(1)
    t_t = torch.tensor(T_flat, dtype=torch.float32).unsqueeze(1)
    
    # Evaluar Solucion Exacta
    Ez_exact, _, _ = exact_solution(x_t, y_t, t_t)
    Ez_exact_mesh = Ez_exact.numpy().reshape(grid_pts, grid_pts)
    
    # Evaluar Prediccion
    inputs = torch.cat([x_t, y_t, t_t], dim=1)
    with torch.no_grad():
        preds = model(inputs)
    Ez_pred_mesh = preds[:, 0:1].numpy().reshape(grid_pts, grid_pts)
    
    # Graficar
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin, vmax = -1.0, 1.0 # Limite de amplitud Ez E0=1
    
    im0 = axes[0].imshow(Ez_exact_mesh, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Solución Exacta (Ez) t={t_snapshot:.2f}")
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(Ez_pred_mesh, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Predicción PINN (Ez) t={t_snapshot:.2f}")
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    fig.colorbar(im1, ax=axes[1])
    
    error = np.abs(Ez_pred_mesh - Ez_exact_mesh)
    im2 = axes[2].imshow(error, extent=[0, L, 0, L], origin='lower', cmap='magma')
    axes[2].set_title("Error Absoluto")
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
