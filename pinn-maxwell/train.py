import os
import torch
import torch.optim as optim
from model import FCN
from sampling import sample_interior, sample_boundary, sample_initial
from physics import total_loss


def train_adam(model, n_epochs=20000, lr=1e-3,
               N_col=20000, N_bc=600, N_ic=8000,
               lambda_pde=1.0, lambda_bc=30.0, lambda_ic=30.0,
               device='cpu', verbose_every=1000):
    """
    Rutina de entrenamiento empleando el optimizador estocástico Adam.

    Mejoras sobre la versión inicial:

    1. CosineAnnealingWarmRestarts (reemplaza StepLR):
       El StepLR reducía el LR abruptamente cada 3k epochs, colapsando
       prematuramente la exploración del landscape no-convexo. El coseno
       oscilante con reinicios periódicos (T_0=5000, T_mult=2) permite
       escapar de mínimos locales y refinarse progresivamente.

    2. Gradient Clipping (max_norm=1.0):
       Las derivadas de segundo orden necesarias para los residuales de
       Maxwell (∂²Ez/∂t², etc.) pueden provocar explosiones de gradiente.
       Clipear la norma máxima estabiliza el backprop sin truncar la
       dirección del gradiente.

    3. Dynamic Lambda Warm-up (Currículo Físico):
       Epochs 1 → WARMUP_END (5000): λ_bc=50, λ_ic=50.
       La red aprende primero a respetar las condiciones iniciales y de
       frontera (IC/BC) antes de intentar satisfacer la EDP interior.
       Esto evita la "solución trivial" (Ez=0 en todo el dominio) que
       el optimizador descubre si BC/IC no están bien ancladas.
       Post-warmup: se usan los lambdas pasados como argumento.

    4. Periodic BC/IC Resampling (cada 3000 epochs):
       Los puntos estáticos de BC e IC pueden provocar overfitting al
       conjunto fijo de frontera. Regenerarlos periódicamente mantiene
       la generalización espacial de las condiciones de borde.

    5. Épocas por defecto aumentadas: 10k → 20k.
       Los modos sinusoidales complejos requieren más iteraciones para
       convergencia estable, especialmente con el warm-up curricular.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # CosineAnnealingWarmRestarts: decaimiento suave con reinicios periódicos
    # T_0=5000 → primer reinicio a los 5k epochs
    # T_mult=2 → cada reinicio es el doble de largo que el anterior
    # eta_min=1e-6 → LR mínimo absoluto al fondo del coseno
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=4000, T_mult=2, eta_min=1e-6
    )

    # 1. Puntos Estáticos iniciales (Frontera e Instante Inicial)
    boundary_points = sample_boundary(N_per_side=N_bc, device=device)
    x_ic, y_ic, t_ic = sample_initial(N=N_ic, device=device)

    history = []

    # Warm-up curricular: pesos BC e IC altos al inicio para anclar las condiciones físicas
    WARMUP_END = 8000
    lambda_bc_warmup = 100.0
    lambda_ic_warmup = 100.0

    for epoch in range(1, n_epochs + 1):

        # --- Dynamic Lambda Schedule (Currículo Físico) ---
        # Fase 1 (warm-up): anclar fuerte las condiciones de frontera e iniciales
        # Fase 2 (entrenamiento): permitir que el PDE domine gradualmente
        if epoch <= WARMUP_END:
            lbc = lambda_bc_warmup
            lic = lambda_ic_warmup
        else:
            lbc = lambda_bc
            lic = lambda_ic

        # --- Periodic BC/IC Resampling ---
        # Regenerar puntos de frontera e IC cada 3000 epochs para evitar
        # overfitting a la distribución fija initial de puntos
        if epoch > 1 and epoch % 3000 == 0:
            boundary_points = sample_boundary(N_per_side=N_bc, device=device)
            x_ic, y_ic, t_ic = sample_initial(N=N_ic, device=device)

        # 2. Puntos Dinámicos (Remuestreo Interior en cada epoch)
        x_col, y_col, t_col = sample_interior(N=N_col, device=device)

        # 3. Flujo Computacional
        optimizer.zero_grad()
        L_tot, L_p, L_b, L_i, r_dict = total_loss(
            model, x_col, y_col, t_col,
            boundary_points, x_ic, y_ic, t_ic,
            lambda_pde, lbc, lic
        )
        L_tot.backward()

        # Gradient Clipping: previene explosiones en derivadas de alto orden del PDE
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

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
            'res_gauss_b': r_dict['gauss_b'],
            'lr': optimizer.param_groups[0]['lr']
        }
        history.append(hist_entry)

        if epoch % verbose_every == 0 or epoch == 1:
            phase = "WARMUP" if epoch <= WARMUP_END else "TRAIN "
            lr_curr = optimizer.param_groups[0]['lr']
            print(f"[{phase} {epoch:05d}] Total: {L_tot.item():.2e} | "
                  f"PDE: {L_p.item():.1e} | BC: {L_b.item():.1e} | IC: {L_i.item():.1e} | "
                  f"LR: {lr_curr:.2e} | lam_BC={lbc:.0f}")
            print(f"  Residuales -> Faraday_x: {r_dict['faraday_x']:.1e} | "
                  f"Faraday_y: {r_dict['faraday_y']:.1e} | "
                  f"Ampere: {r_dict['ampere']:.1e} | Gauss_B: {r_dict['gauss_b']:.1e}")

    return model, history


def train_lbfgs(model, max_iter=3000,
                N_col=20000, N_bc=600, N_ic=8000,
                lambda_pde=1.0, lambda_bc=30.0, lambda_ic=30.0,
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

    Mejoras:
    - max_iter aumentado: 1000 → 3000 para mayor refinamiento post-Adam.
    - history_size aumentado: 50 → 100 para mejor aproximación del Hessiano.
    - Gradient clipping aplicado también dentro del closure para consistencia.
    """
    model.to(device)
    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=max_iter,
        max_eval=int(max_iter * 1.25),
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        history_size=150,        # Aumentado a 150 para Hessiano más preciso
        line_search_fn='strong_wolfe'
    )

    # Datos estáticos para consistencia del Hessiano en todas las evaluaciones del closure
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
        # Gradient clipping también en L-BFGS para evitar pasos destructivos
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        final_loss_val = L_tot.item()
        return L_tot

    optimizer.step(closure)
    return model, final_loss_val


def train_full(model, adam_epochs=20000, lbfgs_iter=3000,
               device='cpu', save_path='results/maxwell_pinn.pth'):
    """
    Aglutinador estático secuencial de optimización de pesos y biases Híbrido.
    (Adam → L-BFGS → Guardado .pth)
    """
    print("===================================================================")
    print("         INICIANDO FASE 1: ENTRENAMIENTO ESTOCÁSTICO ADAM        ")
    print("===================================================================")
    model, history_adam = train_adam(
        model, n_epochs=adam_epochs, lr=1e-3,
        device=device, verbose_every=max(1, adam_epochs // 20)
    )

    print("\n===================================================================")
    print("           INICIANDO FASE 2: REFINAMIENTO HESSIÁN L-BFGS         ")
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
                    layers=[3, 128, 128, 128, 128, 3], device='cpu',
                    fourier_features=True, n_fourier=32, sigma=1.0):
    """
    Restaura las interconexiones en una arquitectura FCN mediante pesos generados y
    fijados dictaminando evaluation mode.

    IMPORTANTE: Los parámetros fourier_features, n_fourier y sigma deben coincidir
    exactamente con los usados al guardar el checkpoint. El buffer B es determinístico
    (misma semilla) gracias a set_seed() en main.py, garantizando reproducibilidad.
    """
    model = FCN(layers, fourier_features=fourier_features,
                n_fourier=n_fourier, sigma=sigma)
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
    print("EJECUCIÓN DE MICRO-PRUEBA (DRY RUN) con Fourier Features + Mejoras de Entrenamiento")
    net = FCN([3, 64, 64, 64, 3], fourier_features=True, n_fourier=16)
    try:
        train_full(net, adam_epochs=200, lbfgs_iter=50, device='cpu',
                   save_path='results/test_maxwell_pinn.pth')
        print("Micro-entrenamiento compiló e iteró fluidamente en Python.")
    except Exception as e:
        print(f"Excepción arrojada: {e}")
