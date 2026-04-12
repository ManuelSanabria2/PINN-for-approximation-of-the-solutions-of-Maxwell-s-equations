import torch
import torch.optim as optim
from physics import compute_physics_residuals
from sampling import sample_collocation_points, sample_initial_conditions, sample_boundary_conditions
from analytical import exact_solution, T_res

def calc_loss(model, N_colloc, N_ic, N_bc, T_max):
    """
    Calcula simultáneamente el Loss de la PDE, de las condiciones iniciales y de frontera.
    """
    # 1. Pérdida Física (Loss PDE)
    x_c, y_c, t_c = sample_collocation_points(N_colloc, T_max)
    res_fx, res_fy, res_amp, res_gauss = compute_physics_residuals(model, x_c, y_c, t_c)
    
    loss_pde = torch.mean(res_fx**2) + torch.mean(res_fy**2) + \
               torch.mean(res_amp**2) + torch.mean(res_gauss**2)
               
    # 2. Pérdida de Condiciones Iniciales (Loss IC)
    x_ic, y_ic, t_ic = sample_initial_conditions(N_ic)
    Ez_ic_exact, Bx_ic_exact, By_ic_exact = exact_solution(x_ic, y_ic, t_ic)
    
    preds_ic = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
    Ez_ic, Bx_ic, By_ic = preds_ic[:, 0:1], preds_ic[:, 1:2], preds_ic[:, 2:3]
    
    loss_ic = torch.mean((Ez_ic - Ez_ic_exact)**2) + \
              torch.mean((Bx_ic - Bx_ic_exact)**2) + \
              torch.mean((By_ic - By_ic_exact)**2)
              
    # 3. Pérdida de Condiciones de Frontera (Loss BC) - Paredes PEC
    x_bc, y_bc, t_bc = sample_boundary_conditions(N_bc, T_max)
    preds_bc = model(torch.cat([x_bc, y_bc, t_bc], dim=1))
    Ez_bc = preds_bc[:, 0:1]
    
    # En PEC la compomente tangencial E es nula. 
    # Ez apunta en el eje Z paralelo a todos los muros X-Y, por lo que Ez=0 siempre (modo TM).
    loss_bc = torch.mean(Ez_bc**2)
    
    # Sumatoria agregada
    # Las penalidades hyperparamétricas se pueden escalar más adelante (λ)
    return loss_pde + loss_ic + loss_bc

def train_model(model, epochs_adam=1500, epochs_lbfgs=500, N_colloc=10000, N_ic=2000, N_bc=2000):
    """
    Entrena el modelo PINN con un híbrido ADAM-BFGS 
    """
    T_max = 2.0 * T_res  # 2 periodos
    
    # --- Optimización ADAM ---
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    
    print("--- Comenzando optimización con ADAM ---")
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        loss = calc_loss(model, N_colloc, N_ic, N_bc, T_max)
        loss.backward()
        optimizer_adam.step()
        
        if epoch % 300 == 0:
            print(f"Adam Epoch {epoch}: Loss = {loss.item():.6e}")
            
    # --- Optimización L-BFGS ---
    print("--- Comenzando optimización L-BFGS ---")
    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=epochs_lbfgs, 
        max_eval=epochs_lbfgs, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        history_size=50, 
        line_search_fn='strong_wolfe'
    )
    
    # L-BFGS requiere un closure func
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = calc_loss(model, N_colloc, N_ic, N_bc, T_max)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    
    eval_loss = calc_loss(model, N_colloc, N_ic, N_bc, T_max)
    print(f"L-BFGS Terminado. Entrenamiento exitosamente completado con Loss final = {eval_loss.item():.6e}")
    
    return model
