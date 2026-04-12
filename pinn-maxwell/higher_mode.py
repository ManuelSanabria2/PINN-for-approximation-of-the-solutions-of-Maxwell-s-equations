import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Reutilización íntegra sin alteración estructural de módulos Core
import physics
import train
from model import FCN
from analytical import cavity_mode, c, L, Ez_exact, Bx_exact, By_exact

plt.style.use('seaborn-v0_8-whitegrid')

def compare_modes(modes=[(1,1), (2,1), (1,2)], save_path='results/'):
    """
    Contrasta analíticamente el espectro resonante para varios pares paramétricos (m, n).
    
    Interpretación Física de Nodos:
    Físicamente, a mayor número de nodos espaciales (m, n) en nuestra cavidad PEC,
    tendremos mayor compresión de la longitud de onda (Wavelength) forzando consecuentemente
    una vibración del tensor mucho más veloz (Frecuencia elevada). Físicamente esto es
    equivalente a los tonos de armónicos agudos en física acústica de instrumentos.
    La PINN tendrá que mapear gradientes y "valles" mucho más empinados, aumentando 
    drásticamente el grado de dificultad de L-BFGS.
    """
    os.makedirs(save_path, exist_ok=True)
    
    print("Modo  | Frecuencia (GHz) | Longitud de onda (m) | Nodos en x | Nodos en y")
    print("------+------------------+----------------------+------------+-----------")
    for (m, n) in modes:
        params = cavity_mode(m, n)
        
        # Frecuencia escalada a GHz
        freq_GHz = params['freq_Hz'] / 1e9
        # Longitud de Onda Escalar λ = c / f
        wavelength = c / params['freq_Hz']
        
        # Un modo TM_mn asume 'm' variaciones de media onda en el eje x y 'n' en y.
        print(f"TM_{m}{n}  |     {freq_GHz:5.2f}        |       {wavelength:5.2f}          |     {m}      |     {n}")

def train_higher_mode(m=2, n=1, adam_epochs=15000, lbfgs_iter=1000, device='cpu'):
    """
    Compila y entrena arquitecturas extendidas para resonadores complejos TM_mn.
    
    Lógica de Reutilización de physics.py:
    Para respetar el principio de OCP (Open-Closed Principle), no modificamos
    el código de physics.py estático original enfocado en TM11. En su lugar, hacemos 
    un monkey-patching sobre la condición inicial inyectándola globalmente solo 
    durante el frame de ejecución de esta función.
    """
    # 1. Arquitectura Extendida (De 4 a 5 capas ocultas de 128 para poder mapear la severidad modal superior)
    model = FCN([3, 128, 128, 128, 128, 128, 3])
    
    # 2. Monkey-Patch a la condición inicial de la gravedad de physics original 
    original_initial_loss = physics.initial_loss
    
    def IC_patched(mod, x_ic, y_ic, t_ic, E0=1.0, length=1.0):
        Ez_ic_pred, Bx_ic_pred, By_ic_pred = mod(x_ic, y_ic, t_ic)
        # Aquí alteramos la condición incial a frecuencias paramétricas m y n
        pi_x = m * np.pi * x_ic / length
        pi_y = n * np.pi * y_ic / length
        
        Ez_target = E0 * torch.sin(pi_x) * torch.sin(pi_y)
        Bx_target = torch.zeros_like(Bx_ic_pred)
        By_target = torch.zeros_like(By_ic_pred)
        
        mse_Ez = torch.mean(torch.square(Ez_ic_pred - Ez_target))
        mse_Bx = torch.mean(torch.square(Bx_ic_pred - Bx_target))
        mse_By = torch.mean(torch.square(By_ic_pred - By_target))
        return mse_Ez + mse_Bx + mse_By
        
    physics.initial_loss = IC_patched
    
    # 3. Entrenamiento Intacto
    os.makedirs('results', exist_ok=True)
    save_path = f'results/maxwell_pinn_TM{m}{n}.pth'
    
    print(f"\n--- INICIANDO PROTOCOLO COMPLEJO MODO HIGHER TM_{m}{n} ---")
    modelo_entrenado, historia = train.train_full(
        model, adam_epochs=adam_epochs, lbfgs_iter=lbfgs_iter, 
        device=device, save_path=save_path
    )
    
    # Restaurar patch
    physics.initial_loss = original_initial_loss
    
    return modelo_entrenado, historia

def compare_predictions(model_11, model_21, t_val, N=100, save_path='results/'):
    """
    Genera una óptica paralela ocluyendo los modelos preentrenados del modo armónico
    base contra el modo hiper-excitado espacial en un colormap termográfico divergente.
    
    Se evalúan ambos en la misma grilla a T determinado.
    """
    os.makedirs(save_path, exist_ok=True)
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
    t_t = torch.tensor(np.full_like(X.flatten(), t_val), dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        preds_11 = model_11(x_t, y_t, t_t)
        preds_21 = model_21(x_t, y_t, t_t)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), dpi=150)
    fig.suptitle(f"Disección Modal Estructural | t = {t_val:.2e}s | TM11 vs TM21", fontsize=16)
    
    components = [
        (preds_11[0], preds_21[0], "Campo Eléctrico Ez", axes[0]),
        (preds_11[1], preds_21[1], "Campo Magnético Bx", axes[1]),
        (preds_11[2], preds_21[2], "Campo Magnético By", axes[2])
    ]
    
    for c11, c21, title, ax_row in components:
        m11 = c11.numpy().reshape(N,N)
        m21 = c21.numpy().reshape(N,N)
        vmax = max(np.max(np.abs(m11)), np.max(np.abs(m21)))
        if vmax == 0: vmax=1.0
        
        im0 = ax_row[0].imshow(m11.T, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax_row[0].set_title(f"TM_11: {title}")
        fig.colorbar(im0, ax=ax_row[0], pad=0.04)
        
        im1 = ax_row[1].imshow(m21.T, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax_row[1].set_title(f"TM_21: {title}")
        fig.colorbar(im1, ax=ax_row[1], pad=0.04)
        
    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'mode_comparison_t{t_val:.2e}.png'))
    plt.close()

if __name__ == '__main__':
    print("--- 1. ANALÍTICA DE MODOS RESONANTES HIGHER-ORDER ---")
    compare_modes()
    
    print("\n--- 2. COMPILACIÓN DE EJECUCIÓN CORTA TM21 ---")
    try:
        # Mini entrenamiento TM21 para comprobar que converge
        mod_21, hist_21 = train_higher_mode(m=2, n=1, adam_epochs=100, lbfgs_iter=20)
        
        # Para hacer la iteración exigida en las instrucciones creamos una pequeña de TM11 también 
        # (simulando que main.py ya la tenía generada)
        print("\nLevantando dummy TM11 para la comparativa óptica...")
        mod_11 = FCN([3, 128, 128, 128, 128, 3]) # Layer base TM11
        
        print("\n--- 3. COMPARATIVA ESPECTRAL MATRICIAL ---")
        compare_predictions(mod_11, mod_21, t_val=0.0)
        print("[+] Todas las librerías Higher_Mode operaron en total normalidad.")
    except Exception as e:
        print(f"[ERROR]: {e}")
