import os
import torch
import torch.optim as optim
from model import FCN
from sampling import sample_interior, sample_boundary, sample_initial
from physics import total_loss

def train_adam(model, n_epochs=10000, lr=1e-3,
               N_col=5000, N_bc=200, N_ic=3000,
               lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
               device='cpu', verbose_every=1000):
    """
    Rutina de entrenamiento empleando el optimizador estocástico Adam.
    Muestrea las penalizaciones perimetrales (BC) y de estado basal (IC) una sola
    vez como fijaciones rígidas al inicio para reducir costo computacional,
    y remuestrea aleatoriamente el interior (colloc points) dinámicamente en cada step
    para prevenir el overfitting de la EDP.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 1. Puntos Estáticos (Frontera e Instante Incial)
    # Se obtienen una única vez antes del bucle para ahorrar coste de tensores.
    boundary_points = sample_boundary(N_per_side=N_bc, device=device)
    x_ic, y_ic, t_ic = sample_initial(N=N_ic, device=device)
    
    history = []
    
    for epoch in range(1, n_epochs + 1):
        # 2. Puntos Dinámicos (Remuestreo Interior)
        x_col, y_col, t_col = sample_interior(N=N_col, device=device)
        
        # 3. Flujo Computacional
        optimizer.zero_grad()
        L_tot, L_p, L_b, L_i, r_dict = total_loss(
            model, x_col, y_col, t_col,
            boundary_points, x_ic, y_ic, t_ic,
            lambda_pde, lambda_bc, lambda_ic
        )
        L_tot.backward()
        optimizer.step()
        
        # 4. Registro y Métrica
        hist_entry = {
            'epoch': epoch,
            'loss_total': L_tot.item(),
            'loss_pde': L_p.item(),
            'loss_bc': L_b.item(),
            'loss_ic': L_i.item(),
            'res_faraday_x': r_dict['faraday_x'],
            'res_faraday_y': r_dict['faraday_y'],
            'res_ampere': r_dict['ampere'],
            'res_gauss_b': r_dict['gauss_b']
        }
        history.append(hist_entry)
        
        if epoch % verbose_every == 0 or epoch == 1:
            print(f"[Epoch {epoch:05d}] Total: {L_tot.item():.2e} | PDE: {L_p.item():.1e} | BC: {L_b.item():.1e} | IC: {L_i.item():.1e}")
            print(f" Residuales -> Faraday_x: {r_dict['faraday_x']:.1e} | Faraday_y: {r_dict['faraday_y']:.1e} | "
                  f"Ampère: {r_dict['ampere']:.1e} | Gauss_B: {r_dict['gauss_b']:.1e}")

    return model, history

def train_lbfgs(model, max_iter=1000,
                N_col=5000, N_bc=200, N_ic=3000,
                lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0,
                device='cpu'):
    """
    Optimización L-BFGS orientada a refinar el error post-Adam.
    
    Reflexión Científica PINN:
        El método L-BFGS (quasi-Newton) es excelente para dinámicas topológicas puramente
        estáticas. Sin embargo, en un problema paramétricamente acoplado al 'Espacio-Tiempo'
        las oscilaciones temporales agudizan el 'Loss Landscape', haciéndolo supremamente
        no convexo, poblado de mesetas locales y dimensiones extendidas. L-BFGS calculará un
        Hessiano aproximado enorme pudiendo estancarse prematuramente si el guess inicial (Adam)
        no fue suficientemente asintótico.
    """
    model.to(device)
    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=max_iter,
        max_eval=max_iter * 1.25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
        line_search_fn='strong_wolfe'
    )
    
    # Evaluar la constancia en closure asume data estática para evitar distorsiones del Hessiano
    x_col, y_col, t_col = sample_interior(N=N_col, device=device)
    boundary_points = sample_boundary(N_per_side=N_bc, device=device)
    x_ic, y_ic, t_ic = sample_initial(N=N_ic, device=device)
    
    final_loss_val = 0.0

    def closure():
        nonlocal final_loss_val
        optimizer.zero_grad()
        L_tot, _, _, _, _ = total_loss(
            model, x_col, y_col, t_col,
            boundary_points, x_ic, y_ic, t_ic,
            lambda_pde, lambda_bc, lambda_ic
        )
        L_tot.backward()
        final_loss_val = L_tot.item()
        return L_tot

    optimizer.step(closure)
    return model, final_loss_val

def train_full(model, adam_epochs=10000, lbfgs_iter=1000,
               device='cpu', save_path='results/maxwell_pinn.pth'):
    """
    Aglutinador estático secuencial de optimización de pesos y biases Híbrido.
    (Adam -> L-BFGS -> Guardado .pth)
    """
    print("===================================================================")
    print("         INICIANDO FASE 1: ENTRENAMIENTO ESTOCÁSTICO ADAM        ")
    print("===================================================================")
    model, history_adam = train_adam(
        model, n_epochs=adam_epochs, lr=1e-3,
        device=device, verbose_every=max(1, adam_epochs // 10)
    )
    
    print("\n===================================================================")
    print("           INICIANDO FASE 2: REFINAMIENTO HESSÍAN L-BFGS         ")
    print("===================================================================")
    model, lbfgs_loss = train_lbfgs(model, max_iter=lbfgs_iter, device=device)
    print(f"[+] L-BFGS Completo. Pérdida Global Consolidada: {lbfgs_loss:.4e}")
    print("===================================================================\n")
    
    # Extracción Opcional y Guardado de Red Resultante
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Estado de la red correctamente alojado en: {save_path}")
        
    return model, history_adam

def load_checkpoint(path='results/maxwell_pinn.pth',
                    layers=[3, 128, 128, 128, 128, 3], device='cpu'):
    """
    Restaura las interconexiones en una arquitectura FCN mediante pesos generados y
    fijados dictaminando evaluation mode.
    """
    model = FCN(layers)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"Pesos de red FCN abstraídos satisfactoriamente desde {path}")
    else:
        print(f"Advertencia: Archivo local no encontrado en '{path}'. Creado con pesos aleatorios.")
    return model

if __name__ == '__main__':
    # Micro-test de validación secuencial del compilador abstracto
    print("EJECUCIÓN DE MICRO-PRUEBA (DRY RUN)")
    net = FCN([3, 64, 64, 64, 3])
    # Tiempos extremadamente bajos de consolidación para testear sin carga asimétrica
    try:
        train_full(net, adam_epochs=200, lbfgs_iter=50, device='cpu', 
                   save_path='results/test_maxwell_pinn.pth')
        print("Micro-entrenamiento compiló e iteró fluidamente en Python.")
    except Exception as e:
        print(f"Excepción arrojada: {e}")
