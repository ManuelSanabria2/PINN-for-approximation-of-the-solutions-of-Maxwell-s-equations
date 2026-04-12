import torch
import numpy as np
from model import FCN
from sampling import sample_interior, sample_boundary, sample_initial

# Constantes Físicas del Vacío en unidades del SI
c = 3e8                           # Velocidad de la luz [m/s]
mu0 = 4 * np.pi * 1e-7            # Permeabilidad magnética del vacío [H/m]
eps0 = 1 / (mu0 * c**2)           # Permitividad eléctrica del vacío [F/m]
L = 1.0                           # Longitud de la cavidad PEC [m]

def compute_fields_and_derivatives(model, x, y, t):
    """
    Computa el paso Forward de la PINN para derivar los 3 campos electromagnéticos
    (Ez, Bx, By) y generar todas las 9 derivadas parciales espaciotemporales.
    
    Eficiencia e Implementación:
    -----------------------------------------------------
    Extraer todos los gradientes simultáneamente tras una única llamada `forward`
    y agruparlos en esta función central es numéricamente eficiente.
    Si derivamos separadamente dentro de cada residual de Maxwell en distintas líneas,
    incurriremos inadvertidamente en múltiples pases inútiles del Backpropagation
    o colisiones en el grafo computacional dinámico.
    """
    # 1. Forward Pass único
    Ez, Bx, By = model(x, y, t)
    
    # 2. Función helper para Autodiferenciación
    def autograd_grad(outputs, inputs):
        return torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,   # Mantener grafo para poder derivar el error retropropragado (loss)
            retain_graph=True
        )[0]
    
    # 3. Derivadas Parciales (Componente Ez)
    dEz_dx = autograd_grad(Ez, x)
    dEz_dy = autograd_grad(Ez, y)
    dEz_dt = autograd_grad(Ez, t)
    
    # 4. Derivadas Parciales (Componente Bx)
    dBx_dx = autograd_grad(Bx, x)
    dBx_dy = autograd_grad(Bx, y)
    dBx_dt = autograd_grad(Bx, t)
    
    # 5. Derivadas Parciales (Componente By)
    dBy_dx = autograd_grad(By, x)
    dBy_dy = autograd_grad(By, y)
    dBy_dt = autograd_grad(By, t)
    
    # 6. Empaquetamiento
    return {
        'fields': (Ez, Bx, By),
        'dEz_dx': dEz_dx, 'dEz_dy': dEz_dy, 'dEz_dt': dEz_dt,
        'dBx_dx': dBx_dx, 'dBx_dy': dBx_dy, 'dBx_dt': dBx_dt,
        'dBy_dx': dBy_dx, 'dBy_dy': dBy_dy, 'dBy_dt': dBy_dt,
    }

def maxwell_residuals(model, x, y, t):
    """
    Ecuaciones Diferenciales Parciales del electromagnetismo encapsuladas.
    Retorna el residual numérico (cuanto se aleja el sistema de dar cero exacto).
    """
    derivs = compute_fields_and_derivatives(model, x, y, t)
    
    # 1. Faraday (Componente x)
    # Ecuación completa: ∇×E = -∂B/∂t
    # En 2D TM (Ez en z; Bx, By en xy), la componente X de ∇×E es ∂Ez/∂y.
    # Por tanto: ∂Ez/∂y = -∂Bx/∂t  --->  R1: ∂Bx/∂t + ∂Ez/∂y = 0
    faraday_x = derivs['dBx_dt'] + derivs['dEz_dy']
    
    # 2. Faraday (Componente y)
    # Ecuación completa: ∇×E = -∂B/∂t
    # Componente Y del rotacional de E es -∂Ez/∂x.
    # Por tanto: -∂Ez/∂x = -∂By/∂t --->  R2: ∂By/∂t - ∂Ez/∂x = 0
    faraday_y = derivs['dBy_dt'] - derivs['dEz_dx']
    
    # 3. Ley de Ampère-Maxwell (Componente z)
    # Ecuación completa: ∇×B = μ0*J + μ0*ε0*(∂E/∂t)
    # Sin cargas ni corrientes (J=0): ∇×B = μ0*ε0*(∂E/∂t)
    # Componente Z del rotacional de B en 2D es (∂By/∂x - ∂Bx/∂y).
    # Por tanto: μ0*ε0*∂Ez/∂t = ∂By/∂x - ∂Bx/∂y ---> R3: μ0*ε0*∂Ez/∂t - (∂By/∂x - ∂Bx/∂y) = 0
    ampere = (mu0 * eps0 * derivs['dEz_dt']) - (derivs['dBy_dx'] - derivs['dBx_dy'])
    
    # 4. Ley de Gauss para el magnetismo
    # Ecuación completa: ∇·B = 0
    # En 2D (xy), la divergencia es espacial ---> R4: ∂Bx/∂x + ∂By/∂y = 0
    gauss_b = derivs['dBx_dx'] + derivs['dBy_dy']
    
    return {
        'faraday_x': faraday_x,
        'faraday_y': faraday_y,
        'ampere': ampere,
        'gauss_b': gauss_b
    }

def boundary_loss(model, boundary_points):
    """
    Penaliza la ruptura de barreras físicas.
    Perfect Electric Conductor (PEC) dictamina Ez = 0 en los 4 bordes.
    Bx y By son validados intrínsecamente usando autograd de las ecuaciones en Gauss B.
    """
    b_loss = 0.0
    for side in ['left', 'right', 'bottom', 'top']:
        x_b, y_b, t_b = boundary_points[side]
        Ez_b, Bx_b, By_b = model(x_b, y_b, t_b)
        
        # MSE(Ez_predicho, 0)
        b_loss += torch.mean(torch.square(Ez_b - 0))
        
    return b_loss

def initial_loss(model, x_ic, y_ic, t_ic, E0=1.0, L=1.0):
    """
    Inyecta la resonancia de arranque (ground truth al instante universal t=0).
    Sujeto a modo TM11:  Ez = E0 * sin(πx/L) * sin(πy/L)
    """
    Ez_ic_pred, Bx_ic_pred, By_ic_pred = model(x_ic, y_ic, t_ic)
    
    # Target matemático 
    pi_x = np.pi * x_ic / L
    pi_y = np.pi * y_ic / L
    Ez_target = E0 * torch.sin(pi_x) * torch.sin(pi_y)
    Bx_target = torch.zeros_like(Bx_ic_pred)
    By_target = torch.zeros_like(By_ic_pred)
    
    mse_Ez = torch.mean(torch.square(Ez_ic_pred - Ez_target))
    mse_Bx = torch.mean(torch.square(Bx_ic_pred - Bx_target))
    mse_By = torch.mean(torch.square(By_ic_pred - By_target))
    
    return mse_Ez + mse_Bx + mse_By

def total_loss(model, x_col, y_col, t_col, boundary_points, x_ic, y_ic, t_ic,
               lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
    """
    Asimila y concentra los 3 grandes macro-errores que rigen a la red, ponderándolos.
    
    ¿Por qué ponderar agresivamente BC e IC (10.0)?
    Las condiciones de contorno e iniciales son formalmente restricciones matemáticas duras;
    si la red neuronal no respeta rigurosamente dónde inician los fotones (IC) o donde 
    rebotan (PEC, BC), derivar el comportamiento infinito vacío del dominio completo
    tendrá nulo sentido matemático en los PDE. Anclar fuerte las fronteras dirige al autograd.
    """
    # 1. Error Interior (Physical Residuals PDE)
    res_dict = maxwell_residuals(model, x_col, y_col, t_col)
    mse_faraday_x = torch.mean(torch.square(res_dict['faraday_x']))
    mse_faraday_y = torch.mean(torch.square(res_dict['faraday_y']))
    mse_ampere = torch.mean(torch.square(res_dict['ampere']))
    mse_gauss = torch.mean(torch.square(res_dict['gauss_b']))
    
    loss_pde = mse_faraday_x + mse_faraday_y + mse_ampere + mse_gauss
    
    # 2. Error Frontera (Bordes PEC) y Origen (t=0)
    loss_bc = boundary_loss(model, boundary_points)
    loss_ic = initial_loss(model, x_ic, y_ic, t_ic, E0=1.0, L=L)
    
    # 3. Sumatoria
    L_total = (lambda_pde * loss_pde) + (lambda_bc * loss_bc) + (lambda_ic * loss_ic)
    
    # Pre-formateo individual para debug en terminales o notebooks
    dict_residuals = {
        'faraday_x': mse_faraday_x.item(),
        'faraday_y': mse_faraday_y.item(),
        'ampere': mse_ampere.item(),
        'gauss_b': mse_gauss.item()
    }
    
    return L_total, loss_pde, loss_bc, loss_ic, dict_residuals

if __name__ == '__main__':
    print("--- INICIANDO DIAGNÓSTICO DEL MÓDULO FÍSICO DE MAXWELL ---")
    # Instanciamos modelo random sin entrenar
    test_model = FCN([3, 64, 64, 64, 3])
    
    # Generar muestras
    x_c, y_c, t_c = sample_interior(N=10)
    bp = sample_boundary(N_per_side=10)
    x_i, y_i, t_i = sample_initial(N=10)
    
    # Computar Total Loss con Flujo de Gradientes
    L_t, L_p, L_b, L_i, dict_r = total_loss(test_model, x_c, y_c, t_c, bp, x_i, y_i, t_i)
    
    print("\n[+] Flujo Forward y Backward (Autograd) computado con éxito sin ruptura de tensores.")
    print(f"Total Loss Híbrida: {L_t.item():.6f}")
    print(f" ▸ PDE  Loss:       {L_p.item():.6f}")
    print(f" ▸ BC   Loss:       {L_b.item():.6f}")
    print(f" ▸ IC   Loss:       {L_i.item():.6f}")
    
    print("\nDesglose de los residuales matemáticos (Mean Squared):")
    for key, val in dict_r.items():
        print(f"   · {key:10s} : {val:.6e}")
