import os
import torch
import numpy as np

from analytical import Ez_exact, Bx_exact, By_exact, cavity_mode, L, eps0, mu0
from sampling import sample_interior, sample_boundary, sample_initial
from physics import maxwell_residuals

def l2_error_per_field(model, N=50, N_t=20):
    """
    Evalúa el error relativo L2 porcentual para los 3 campos a lo largo del tiempo.
    """
    params = cavity_mode(1, 1)
    T_max = 2.0 * params['T_period']
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    t_lin = np.linspace(0, T_max, N_t)
    X, Y, Time = np.meshgrid(x_lin, y_lin, t_lin, indexing='ij')
    
    x_t = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
    t_t = torch.tensor(Time.flatten(), dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        preds = model(x_t, y_t, t_t)
        Ez_p, Bx_p, By_p = preds[0], preds[1], preds[2]
        
    Ez_anal = torch.tensor(Ez_exact(X, Y, Time, 1, 1).flatten(), dtype=torch.float32).unsqueeze(1)
    Bx_anal = torch.tensor(Bx_exact(X, Y, Time, 1, 1).flatten(), dtype=torch.float32).unsqueeze(1)
    By_anal = torch.tensor(By_exact(X, Y, Time, 1, 1).flatten(), dtype=torch.float32).unsqueeze(1)
    
    def rel_l2(pred, exact):
        return (torch.linalg.norm(pred - exact, 2) / torch.linalg.norm(exact, 2)).item() * 100.0

    return {
        'Ez': rel_l2(Ez_p, Ez_anal),
        'Bx': rel_l2(Bx_p, Bx_anal),
        'By': rel_l2(By_p, By_anal)
    }

def residual_per_equation(model, N_test=8000):
    """
    Verifica cuánto se "violan" las ecuaciones de Maxwell en puntos de evaluación.
    Requires Grad debe estar activo para permitir la derivación matemática del autograd.
    """
    x_t, y_t, t_t = sample_interior(N=N_test, device='cpu')
    res_dict = maxwell_residuals(model, x_t, y_t, t_t)
    
    out = {}
    for key, val in res_dict.items():
        val_abs = torch.abs(val.detach())
        out[key] = {
            'mean': torch.mean(val_abs).item(),
            'max': torch.max(val_abs).item()
        }
    return out

def boundary_error(model, N_per_side=500):
    """
    Constata que la condición Perfect Electric Conductor (Ez = 0)
    se mantenga sólida a lo largo de todas las fronteras espaciales.
    """
    bp = sample_boundary(N_per_side=N_per_side, device='cpu')
    errs = []
    
    with torch.no_grad():
        for side in ['left', 'right', 'bottom', 'top']:
            x_b, y_b, t_b = bp[side]
            Ez_b, _, _ = model(x_b, y_b, t_b)
            errs.append(torch.abs(Ez_b))
            
    all_errs = torch.cat(errs)
    return {
        'mean': torch.mean(all_errs).item(),
        'max': torch.max(all_errs).item()
    }

def initial_condition_error(model, N=3000):
    """
    Valida las IC de los campos a lo largo de L2.
    """
    x_ic, y_ic, t_ic = sample_initial(N=N, device='cpu')
    
    with torch.no_grad():
        Ez_p, Bx_p, By_p = model(x_ic, y_ic, t_ic)
        
    Ez_anal = torch.tensor(Ez_exact(x_ic.numpy(), y_ic.numpy(), t_ic.numpy(), 1, 1), dtype=torch.float32)
    Bx_anal = torch.zeros_like(Ez_p)
    By_anal = torch.zeros_like(Ez_p)
    
    def l2_norm(pred, exact):
        # Usamos norma absoluta (MSE) en vez de división L2 si la matriz base es todo ceros
        if torch.max(torch.abs(exact)) == 0.0:
            return torch.mean(torch.square(pred - exact)).item()
        return (torch.linalg.norm(pred - exact, 2) / torch.linalg.norm(exact, 2)).item() * 100.0

    return {
        'Ez': l2_norm(Ez_p, Ez_anal),
        'Bx': l2_norm(Bx_p, Bx_anal),
        'By': l2_norm(By_p, By_anal)
    }

def energy_conservation_error(model, N=40, N_t=30):
    """
    Calcula la variación térmica asincrónica (Conservación Energética Térmica)
    U(t) = ∫∫ u(x,y,t) dxdy
    """
    params = cavity_mode(1, 1)
    T_max = 2.0 * params['T_period']
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    t_lin = np.linspace(0, T_max, N_t)
    dx, dy = x_lin[1]-x_lin[0], y_lin[1]-y_lin[0]
    
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
    
    U_arr = []
    
    with torch.no_grad():
        for t_val in t_lin:
            t_flat = torch.full_like(x_flat, t_val)
            preds = model(x_flat, y_flat, t_flat)
            Ez, Bx, By = preds[0].numpy(), preds[1].numpy(), preds[2].numpy()
            
            u_density = 0.5 * eps0 * (Ez**2) + (0.5 / mu0) * (Bx**2 + By**2)
            U = np.sum(u_density) * dx * dy
            U_arr.append(U)
            
    U_arr = np.array(U_arr)
    # Conservación de U: Cuánto varió el error
    U_mean = np.mean(U_arr)
    U_max = np.max(U_arr)
    U_min = np.min(U_arr)
    
    conservation_pct = ((U_max - U_min) / U_mean) * 100.0 if U_mean != 0 else 0.0
    
    return {
        'U_mean': U_mean,
        'U_std': np.std(U_arr),
        'conservation_error_pct': conservation_pct
    }

def full_report(model, save_path='results/'):
    """
    Orquesta y reporta todas las métricas de éxito en una tabla tabular ANSI.
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Ejecución modular
    l2 = l2_error_per_field(model)
    res = residual_per_equation(model)
    ene = energy_conservation_error(model)
    
    # 2. Plantilla Tabular
    report_text = f"""
  ╔═════════════════════════════════════════════════════════════╗
  ║        REPORTE DE VALIDACIÓN — PINN MAXWELL TM₁₁           ║
  ╠══════════════════════════╦══════════════╦═══════════════════╣
  ║ ERRORES POR CAMPO        ║ Valor        ║ Meta              ║
  ╠══════════════════════════╬══════════════╬═══════════════════╣
  ║ Error L2 Ez              ║ {l2['Ez']:>7.2f}%      ║ < 2%              ║
  ║ Error L2 Bx              ║ {l2['Bx']:>7.2f}%      ║ < 2%              ║
  ║ Error L2 By              ║ {l2['By']:>7.2f}%      ║ < 2%              ║
  ╠══════════════════════════╬══════════════╬═══════════════════╣
  ║ RESIDUALES POR ECUACIÓN  ║ Media        ║ Meta              ║
  ╠══════════════════════════╬══════════════╬═══════════════════╣
  ║ Faraday (x)              ║ {res['faraday_x']['mean']:>10.2e}   ║ < 1e-3            ║
  ║ Faraday (y)              ║ {res['faraday_y']['mean']:>10.2e}   ║ < 1e-3            ║
  ║ Ampère-Maxwell           ║ {res['ampere']['mean']:>10.2e}   ║ < 1e-3            ║
  ║ Gauss magnética          ║ {res['gauss_b']['mean']:>10.2e}   ║ < 1e-4            ║
  ╠══════════════════════════╬══════════════╬═══════════════════╣
  ║ Conservación de energía  ║ {ene['conservation_error_pct']:>7.2f}%      ║ < 5%              ║
  ╚══════════════════════════╩══════════════╩═══════════════════╝
"""
    try:
        print(report_text)
    except UnicodeEncodeError:
        print("[!] Tabla generada exitosamente. (Saltando previsualización en consola debido a esquema Unicode)")
    
    file_path = os.path.join(save_path, 'validation_report.txt')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"[+] Reporte exportado a: {file_path}")
    
    return {
        'l2': l2,
        'residuals': res,
        'energy': ene
    }

if __name__ == '__main__':
    from model import FCN
    import torch
    
    print("\nInicializando evaluación sobre arquitectura virgen (Dry Run)...")
    net = FCN([3, 32, 32, 3])
    full_report(net)
