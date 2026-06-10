import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from analytical import Ez_exact, Bx_exact, By_exact, cavity_mode, c, mu0, eps0, L

# Forzar estilo estético científico
plt.style.use('seaborn-v0_8-whitegrid')

def safe_makedirs(save_path):
    os.makedirs(save_path, exist_ok=True)

def get_model_device(model):
    """Detecta el device donde vive el modelo (CPU o CUDA)."""
    return next(model.parameters()).device

def calc_l2_error(pred, exact):
    """Calcula la métrica escalar relativa Norma L2 para títulos gráficos."""
    pred_vec = pred.flatten()
    exact_vec = exact.flatten()
    return np.linalg.norm(pred_vec - exact_vec, 2) / np.linalg.norm(exact_vec, 2)

def plot_fields_snapshot(model, t_val, N=100, save_path='results/'):
    """
    Genera un snapshot 3x3 que disecciona predicho vs teórico vs mapa de error absoluto
    para Ez, Bx, y By simultáneamente en el plano bidimensional en un t dado.
    """
    safe_makedirs(save_path)
    device = get_model_device(model)
    
    # 1. Preparar Grid
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    t_t = torch.tensor(np.full_like(X.flatten(), t_val), dtype=torch.float32).unsqueeze(1).to(device)
    
    # 2. Evaluación del Ground Truth (Analítico)
    Ez_anal = Ez_exact(X, Y, t_val).reshape(N, N)
    Bx_anal = Bx_exact(X, Y, t_val).reshape(N, N)
    By_anal = By_exact(X, Y, t_val).reshape(N, N)
    
    # 3. Evaluación PINN
    with torch.no_grad():
        preds = model(x_t, y_t, t_t)
        # Recordar que FCN retorna lista [Ez, Bx, By]
        Ez_pinn = preds[0].cpu().numpy().reshape(N, N)
        Bx_pinn = preds[1].cpu().numpy().reshape(N, N)
        By_pinn = preds[2].cpu().numpy().reshape(N, N)
        
    # 4. Cálculo Errores L2 Relativo
    err_Ez = calc_l2_error(Ez_pinn, Ez_anal)
    err_Bx = calc_l2_error(Bx_pinn, Bx_anal)
    err_By = calc_l2_error(By_pinn, By_anal)
    
    # Diferencia Absoluta Matriz
    err_abs_Ez = np.abs(Ez_pinn - Ez_anal)
    err_abs_Bx = np.abs(Bx_pinn - Bx_anal)
    err_abs_By = np.abs(By_pinn - By_anal)

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), dpi=150)
    fig.suptitle(f"Campos Electromagnéticos y Error | t = {t_val:.2e}s | "
                 f"L2(Ez): {err_Ez:.2%} | L2(Bx): {err_Bx:.2%} | L2(By): {err_By:.2%}", fontsize=14, y=0.98)
    
    components = [
        (Ez_pinn, Ez_anal, err_abs_Ez, 'Ez', axes[0]),
        (Bx_pinn, Bx_anal, err_abs_Bx, 'Bx', axes[1]),
        (By_pinn, By_anal, err_abs_By, 'By', axes[2])
    ]
    
    for pinn, anal, err, name, ax_row in components:
        vmax = max(np.max(np.abs(anal)), np.max(np.abs(pinn)))
        if vmax == 0: vmax = 1.0 # fallback seguro
        
        # PINN
        im0 = ax_row[0].imshow(pinn.T, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax_row[0].set_title(f"{name} (Predicción PINN)")
        fig.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)
        
        # Analítica
        im1 = ax_row[1].imshow(anal.T, extent=[0, L, 0, L], origin='lower', cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax_row[1].set_title(f"{name} (Analítico Exacto)")
        fig.colorbar(im1, ax=ax_row[1], fraction=0.046, pad=0.04)
        
        # Error Hot
        im2 = ax_row[2].imshow(err.T, extent=[0, L, 0, L], origin='lower', cmap='hot')
        ax_row[2].set_title(f"|Error Absoluto {name}|")
        fig.colorbar(im2, ax=ax_row[2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    output_png = os.path.join(save_path, f'snapshot_t{t_val:.2e}.png')
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()


def plot_time_evolution(model, x_probe=0.5, y_probe=0.5, N_t=200, save_path='results/'):
    """
    Rastrea un sensor topológico en un punto en el espacio (x,y) pero a través del tiempo,
    comparando los armónicos senoidales exactos vs la regresión de la PINN.
    """
    safe_makedirs(save_path)
    device = get_model_device(model)
    
    params = cavity_mode(1, 1)
    T_max = 2.0 * params['T_period']
    t_lin = np.linspace(0, T_max, N_t)
    
    # Convertir a tensores torch
    x_t = torch.full((N_t, 1), x_probe, dtype=torch.float32).to(device)
    y_t = torch.full((N_t, 1), y_probe, dtype=torch.float32).to(device)
    t_t = torch.tensor(t_lin, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        preds = model(x_t, y_t, t_t)
        Ez_pinn = preds[0].cpu().numpy().flatten()
        Bx_pinn = preds[1].cpu().numpy().flatten()
        By_pinn = preds[2].cpu().numpy().flatten()
        
    Ez_anal = Ez_exact(x_probe, y_probe, t_lin, 1, 1)
    Bx_anal = Bx_exact(x_probe, y_probe, t_lin, 1, 1)
    By_anal = By_exact(x_probe, y_probe, t_lin, 1, 1)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=150)
    fig.suptitle(f"Señal y Evolución Temporal (Sensor en x={x_probe}, y={y_probe})", fontsize=14)
    
    data = [
        (Ez_pinn, Ez_anal, 'Ez'),
        (Bx_pinn, Bx_anal, 'Bx'),
        (By_pinn, By_anal, 'By')
    ]
    
    for ax, (pinn, anal, label) in zip(axes, data):
        err_abs = np.abs(pinn - anal)
        
        ax.plot(t_lin, pinn, color='blue', label='PINN Predicción', linewidth=2)
        ax.plot(t_lin, anal, color='red', linestyle='--', label='Analítica', linewidth=2)
        ax.fill_between(t_lin, -err_abs, err_abs, color='green', alpha=0.3, label='|Error Absoluto|')
        
        ax.set_ylabel(f"Amplitud {label}")
        ax.legend(loc='upper right')
        
    axes[2].set_xlabel("Tiempo (s)")
    plt.tight_layout()
    output_png = os.path.join(save_path, 'time_evolution.png')
    plt.savefig(output_png)
    plt.close()


def plot_electromagnetic_energy(model, N=80, N_t=50, save_path='results/'):
    """
    Rastrea y grafica la densidad de energía electromagnética de Maxwell
    y el decaimiento cinético en el tiempo evaluando la conservación fundamental.
    u(x,y,t) = ½*ε₀*|Ez|² + (1/2*μ₀)*(|Bx|² + |By|²)
    """
    safe_makedirs(save_path)
    device = get_model_device(model)
    
    params = cavity_mode(1, 1)
    T = params['T_period']
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    t_lin = np.linspace(0, 2*T, N_t)
    dx, dy = x_lin[1]-x_lin[0], y_lin[1]-y_lin[0]
    
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    
    U_total = []
    
    # 1. Recolección Cronometrizada para Subplot 2 (Energy vs Time)
    for t_step in t_lin:
        with torch.no_grad():
            t_curr = torch.full_like(x_t, t_step)
            preds = model(x_t, y_t, t_curr)
            Ez_i = preds[0].cpu().numpy().flatten()
            Bx_i = preds[1].cpu().numpy().flatten()
            By_i = preds[2].cpu().numpy().flatten()
            
            # Formular la densidad local de la energía (Densidad Escalar en el nodo)
            u_density = 0.5 * eps0 * (Ez_i**2) + (0.5 / mu0) * (Bx_i**2 + By_i**2)
            
            # U total es escalar: Integral bidimensional por sumatoria de Riemann de u(x,y)
            U = np.sum(u_density) * dx * dy
            U_total.append(U)
            
            if t_step == t_lin[0]:
                u_density_t0 = u_density.reshape(N, N)
            dist_to_tq = np.abs(t_step - T*0.25)
            # Guardamos mapeo del snapshot t=T/4 apróx
            if t_step == t_lin[np.argmin(np.abs(t_lin - T*0.25))]:
                u_density_tq = u_density.reshape(N, N)

    # 2. Lienzo Gráfico
    fig = plt.figure(figsize=(15, 6), dpi=150)
    fig.suptitle("Dinámica Termodinámica: Densidad de Energía Electromagnética", fontsize=14)
    
    # -- Snapshot t=0 -- 
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(u_density_t0.T, extent=[0, L, 0, L], origin='lower', cmap='plasma')
    ax1.set_title("u(x,y) a t = 0")
    fig.colorbar(im1, ax=ax1, format='%.1e')
    
    # -- Snapshot t=T/4 --
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(u_density_tq.T, extent=[0, L, 0, L], origin='lower', cmap='plasma')
    ax2.set_title("u(x,y) a t = T/4")
    fig.colorbar(im2, ax=ax2, format='%.1e')

    # -- Gráfico de Conservación (Ley Cero Termo / TEO DE POYNTING) --
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(t_lin, U_total, 'o-', color='purple', linewidth=2)
    ax3.set_ylim([0, max(U_total)*1.2]) # Aseguramos visualización que se mueva o parezca recta
    ax3.set_title("Energía Total $U(t) = \int\int u\;dxdy$ (Conservación)")
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Joules")
    
    plt.tight_layout()
    output_png = os.path.join(save_path, 'electromagnetic_energy.png')
    plt.savefig(output_png)
    plt.close()


def plot_training_history(history, save_path='results/'):
    """
    Toma el arreglo History de Adam y plotea los macro losses por sección y 
    los residuales uninominales por variable para la autocrítica en escala log-y.
    """
    safe_makedirs(save_path)
    if not history:
        print("La historia está vacía. Abortando plot_training_history.")
        return
    
    epochs = [h['epoch'] for h in history]
    l_tot = [h['loss_total'] for h in history]
    l_pde = [h['loss_pde'] for h in history]
    l_bc = [h['loss_bc'] for h in history]
    l_ic = [h['loss_ic'] for h in history]
    
    r_fx = [h['res_faraday_x'] for h in history]
    r_fy = [h['res_faraday_y'] for h in history]
    r_amp = [h['res_ampere'] for h in history]
    r_gaus = [h['res_gauss_b'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # Panel 1: Main Losses Ponderados
    axes[0].plot(epochs, l_tot, color='black', linewidth=2, label='Loss Total')
    axes[0].plot(epochs, l_pde, color='blue', alpha=0.7, label='PDE Loss')
    axes[0].plot(epochs, l_bc, color='red', alpha=0.7, label='Boundary Loss')
    axes[0].plot(epochs, l_ic, color='green', alpha=0.7, label='Initial Loss')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Convergencia del Sistema PINN')
    axes[0].legend()
    
    # Panel 2: Residuales Particulares
    axes[1].plot(epochs, r_fx, color='cyan', alpha=0.8, label='Res Faraday X')
    axes[1].plot(epochs, r_fy, color='teal', alpha=0.8, label='Res Faraday Y')
    axes[1].plot(epochs, r_amp, color='orange', alpha=0.8, label='Res Ampère')
    axes[1].plot(epochs, r_gaus, color='purple', alpha=0.8, label='Res Gauss B')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Residual Error')
    axes[1].set_title('Desglose de Residuales Maxwell (PDE puras)')
    axes[1].legend()

    plt.tight_layout()
    output_png = os.path.join(save_path, 'training_history.png')
    plt.savefig(output_png)
    plt.close()


def plot_field_lines(model, t_val, N=30, save_path='results/'):
    """
    Esboza diagramas vectoriales para mostrar las iteraciones del campo cruzado.
    Streamlines y flujos para el tensor Magnético B.
    """
    safe_makedirs(save_path)
    device = get_model_device(model)
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    X, Y = np.meshgrid(x_lin, y_lin)
    
    # Tensores para pase inferencial
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    t_t = torch.tensor(np.full_like(X.flatten(), t_val), dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        preds = model(x_t, y_t, t_t)
        Ez = preds[0].cpu().numpy().reshape(N, N)
        Bx = preds[1].cpu().numpy().reshape(N, N)
        By = preds[2].cpu().numpy().reshape(N, N)
        
    B_mag = np.sqrt(Bx**2 + By**2)
    if B_mag.max() == 0: B_mag += 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle(f"Campo Vectorial a t = {t_val:.2e}", fontsize=14)
    
    # Subplot 1: Streamlines usando plt.streamplot sobre magnitud
    strm = axes[0].streamplot(X, Y, Bx, By, color=B_mag, cmap='autumn', linewidth=1.5, density=1.5)
    axes[0].set_title("Streamlines Campo Magnético (B)")
    fig.colorbar(strm.lines, ax=axes[0], label='|B|')
    
    # Subplot 2: Campo Base Contorno Ez + Quiver field 
    # El contourf dibuja Ez y quiver superpone las flechas de B
    c1 = axes[1].contourf(X, Y, Ez, levels=30, cmap='RdBu_r', alpha=0.7)
    q = axes[1].quiver(X, Y, Bx, By, width=0.005, pivot='mid', scale_units='xy', scale=np.max(B_mag)*2)
    axes[1].set_title("Superposición Vectorial B sobre Contorno Eléctrico Ez")
    fig.colorbar(c1, ax=axes[1], label='Ez')

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
    plt.tight_layout()
    output_png = os.path.join(save_path, f'field_lines_t{t_val:.2e}.png')
    plt.savefig(output_png)
    plt.close()


def plot_poynting_vector(model, t_val, N=40, save_path='results/'):
    """
    Calcula y grafica el Vector de Poynting S = E x H.
    Punto 2: Demostración de las Leyes de Conservación.
    En TM11: E = (0, 0, Ez), B = (Bx, By, 0).
    H = B / mu0.
    S = E x H = (Ez * Hy, -Ez * Hx, 0)
    """
    safe_makedirs(save_path)
    device = get_model_device(model)
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
    t_t = torch.tensor(np.full_like(X.flatten(), t_val), dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        preds = model(x_t, y_t, t_t)
        Ez = preds[0].cpu().numpy().reshape(N, N)
        Bx = preds[1].cpu().numpy().reshape(N, N)
        By = preds[2].cpu().numpy().reshape(N, N)
        
    # Convertir B a H
    Hx = Bx / mu0
    Hy = By / mu0
    
    # S = E x H
    Sx = Ez * Hy
    Sy = -Ez * Hx
    S_mag = np.sqrt(Sx**2 + Sy**2)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    c = ax.contourf(X, Y, S_mag, levels=30, cmap='viridis')
    q = ax.quiver(X[::2, ::2], Y[::2, ::2], Sx[::2, ::2], Sy[::2, ::2], color='white', alpha=0.8)
    
    ax.set_title(f"Flujo de Energía (Vector de Poynting $\mathbf{{S}}$) | t = {t_val:.2e}s")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(c, ax=ax, label='|S| ($W/m^2$)')
    
    output_png = os.path.join(save_path, f'poynting_t{t_val:.2e}.png')
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()


def plot_spatial_residuals(model, t_val, N=50, save_path='results/'):
    """
    Muestra un mapa espacial de los residuos de las ecuaciones de Maxwell.
    Punto 2: Monitoreo del Residuo Físico.
    """
    from physics import maxwell_residuals
    safe_makedirs(save_path)
    device = get_model_device(model)
    
    x_lin = np.linspace(0.05, 0.95, N)
    y_lin = np.linspace(0.05, 0.95, N)
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
    t_t = torch.tensor(np.full_like(X.flatten(), t_val), dtype=torch.float32).unsqueeze(1).to(device).requires_grad_(True)
    
    res_dict = maxwell_residuals(model, x_t, y_t, t_t)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    fig.suptitle(f"Distribución Espacial del Residuo PDE | t = {t_val:.2e}s", fontsize=14)
    
    equations = [
        (res_dict['faraday_x'], 'Faraday X', axes[0, 0]),
        (res_dict['faraday_y'], 'Faraday Y', axes[0, 1]),
        (res_dict['ampere'], 'Ampère-Maxwell', axes[1, 0]),
        (res_dict['gauss_b'], 'Gauss Magnética', axes[1, 1])
    ]
    
    for res, name, ax in equations:
        r_np = torch.abs(res).detach().cpu().numpy().reshape(N, N)
        im = ax.imshow(r_np.T, extent=[0, L, 0, L], origin='lower', cmap='inferno')
        ax.set_title(f"Residuo {name}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png = os.path.join(save_path, f'spatial_residuals_t{t_val:.2e}.png')
    plt.savefig(output_png)
    plt.close()


if __name__ == '__main__':
    # =========================================================================
    # INSTRUCCIONES DE USO AUTÓNOMO / DEBUGGING GRAFOLÓGICO
    # =========================================================================
    # Este módulo no debe ejecutarse con una red sin entrenamiento ("cruda"), ya 
    # que los gráficos mostrarán distribuciones carentes de significado (ruido Random).
    # Sin embargo, a modo de testeo de flujo:
    
    print("Iniciando prueba unitaria de la librería Matplotlib / Visualización...")
    from model import FCN
    import torch
    dummy_model = FCN([3, 32, 32, 3])
    
    print("Creando Snapshot y Errores (1/5)...")
    plot_fields_snapshot(dummy_model, t_val=0.0)
    
    print("Creando Time Evolution Plot (2/5)...")
    plot_time_evolution(dummy_model)
    
    print("Integrando Energía Electromagnética y Densidad (3/5)...")
    plot_electromagnetic_energy(dummy_model)
    
    print("Graficando Historia Dummy (4/5)...")
    # Generamos un dummy dict array map para no crashear
    h = [{'epoch':i,'loss_total':1/i, 'loss_pde':1/i, 'loss_bc':1/i, 'loss_ic':1/i,
          'res_faraday_x':1/i, 'res_faraday_y':1/i, 'res_ampere':1/i, 'res_gauss_b':1/i} for i in range(1,10)]
    plot_training_history(h)
    
    print("Esbozando Vectores Quiver y Streamlines (5/5)...")
    plot_field_lines(dummy_model, t_val=0.0)
    
    print("[+] Todos los recursos gráficos se renderizaron correctamente y eludieron el RAM leak de Pyplot.")
    print("Búscalos en tu directorio `/results/`")
    # Nota: Si el código corre exitosamente sin memory errors, está verificado el backbend gràfico.
