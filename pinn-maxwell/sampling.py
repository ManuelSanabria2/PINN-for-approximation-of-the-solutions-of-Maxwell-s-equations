import os
import torch
import matplotlib.pyplot as plt

# Constantes del Dominio
L = 1.0           # Longitud de la cavidad PEC en metros
# Si c=1, ky=pi, kx=pi -> omega = pi*sqrt(2). T = 2*pi/omega = sqrt(2). Dos periodos = 2*sqrt(2) = 2.828427
T_MAX = 2.828427  # Cubrimos 2 periodos completos de oscilación

def sample_interior(N=5000, device='cpu'):
    """
    Genera puntos aleatorios basandose en la distribución uniforme dentro del
    dominio matemático abierto del espacio-tiempo.
    
    Anotación crítica:
        Se establece explícitamente "requires_grad=True". Estas muestras pasarán
        a través de nuestra red neuronal FCN y se requiere el grafo para que el módulo
        physics.py pueda construir los operadores de derivadas parciales autodiferenciables
        por medio de Backpropagation (∂/∂x, ∂/∂y, ∂/∂t) del PDE evaluado con Loss.
    """
    # Generando muestras [0, 1] y dimensionando al rango real
    x = torch.rand(N, 1, device=device) * L
    y = torch.rand(N, 1, device=device) * L
    t = torch.rand(N, 1, device=device) * T_MAX
    
    # Activando trackeo de derivadas
    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    
    return x, y, t

def sample_boundary(N_per_side=200, device='cpu'):
    """
    Genera puntos sobre los cuatro bordes espaciales inmersos en cualquier instante 
    t evaluado. Estos puntos formarán parte del dataset Boundary Conditions.
    
    Anotación crítica:
        NO es necesario computar la gradiente sobre estos puntos porque su respectiva
        "Pérdida" calcula simplemente el residuo aritmético respecto a cero (MSE directo).
        
        Además, la condición base 'Ez = 0' en los márgenes de los ejes representa nuestro
        comportamiento modelado para el Perfect Electric Conductor (paredes de la cavidad de PEC).
    """
    # Muro Left: x=0
    x_left = torch.zeros((N_per_side, 1), device=device)
    y_left = torch.rand(N_per_side, 1, device=device) * L
    t_left = torch.rand(N_per_side, 1, device=device) * T_MAX
    
    # Muro Right: x=L
    x_right = torch.full((N_per_side, 1), L, device=device)
    y_right = torch.rand(N_per_side, 1, device=device) * L
    t_right = torch.rand(N_per_side, 1, device=device) * T_MAX
    
    # Muro Bottom: y=0
    x_bot = torch.rand(N_per_side, 1, device=device) * L
    y_bot = torch.zeros((N_per_side, 1), device=device)
    t_bot = torch.rand(N_per_side, 1, device=device) * T_MAX
    
    # Muro Top: y=L
    x_top = torch.rand(N_per_side, 1, device=device) * L
    y_top = torch.full((N_per_side, 1), L, device=device)
    t_top = torch.rand(N_per_side, 1, device=device) * T_MAX
    
    # Diccionario separado explicitando cada muro espacial
    return {
        'left': (x_left, y_left, t_left),
        'right': (x_right, y_right, t_right),
        'bottom': (x_bot, y_bot, t_bot),
        'top': (x_top, y_top, t_top)
    }

def sample_initial(N=3000, device='cpu'):
    """
    Genera puntos geolocalizados exactamente en el instante de origen dimensional t=0.
    Son requeridos para aplicar nuestra Pérdida Basal donde los campos se ajustarán
    a las condiciones numéricas Iniciales (IC): Ez(x,y,0), Bx(x,y,0) y By(x,y,0).
    """
    x = torch.rand(N, 1, device=device) * L
    y = torch.rand(N, 1, device=device) * L
    t_zeros = torch.zeros((N, 1), device=device)
    
    return x, y, t_zeros

def visualize_sampling(save_path='results/'):
    """
    Renderiza y almacena el vector dimensional de todos los grupos muestrales (x,t,y)
    usando subplots representativos de la red.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Muestras simuladas para mantener las formas visibles (escaladas down)
    x_in, y_in, t_in = sample_interior(N=800)
    bc = sample_boundary(N_per_side=150)
    x_ic, y_ic, t_ic = sample_initial(N=400)
    
    # Concatenar boundary tensors para coloreado masivo
    x_b = torch.cat([bc['left'][0], bc['right'][0], bc['bottom'][0], bc['top'][0]])
    y_b = torch.cat([bc['left'][1], bc['right'][1], bc['bottom'][1], bc['top'][1]])
    t_b = torch.cat([bc['left'][2], bc['right'][2], bc['bottom'][2], bc['top'][2]])
    
    # Remoción de gradiente a NUMPY para dibujado a memoria en C
    x_i_np, y_i_np, t_i_np = x_in.detach().numpy(), y_in.detach().numpy(), t_in.detach().numpy()
    x_b_np, y_b_np, t_b_np = x_b.numpy(), y_b.numpy(), t_b.numpy()
    x_ic_np, y_ic_np, t_ic_np = x_ic.numpy(), y_ic.numpy(), t_ic.numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ------------- Subplot 1 (X, Y) -------------
    axes[0].scatter(x_i_np, y_i_np, s=8, alpha=0.3, color='blue', label='Interior Domain')
    axes[0].scatter(x_ic_np, y_ic_np, s=8, alpha=0.7, color='green', label='Initial (t=0)')
    axes[0].scatter(x_b_np, y_b_np, s=8, alpha=0.5, color='red', label='Boundary PEC')
    axes[0].set_xlabel("Eje X (metros)")
    axes[0].set_ylabel("Eje Y (metros)")
    axes[0].set_title("Dominio Espacial: Proyección (X, Y)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # ------------- Subplot 2 (X, T) -------------
    axes[1].scatter(x_i_np, t_i_np, s=8, alpha=0.3, color='blue', label='Interior Domain')
    axes[1].scatter(x_ic_np, t_ic_np, s=20, alpha=1.0, color='green', label='Initial (t=0)')
    axes[1].scatter(x_b_np, t_b_np, s=8, alpha=0.5, color='red', label='Boundary PEC')
    axes[1].set_xlabel("Eje X (metros)")
    axes[1].set_ylabel("Tiempo (segundos)")
    axes[1].set_title("Evolutivo Espacial-Temporal: Proyección (X, T)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_filename = os.path.join(save_path, 'sampling_visualization.png')
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    print(f"\n[+] Documentación gráfica: {output_filename}")


if __name__ == '__main__':
    # 1. Dibujamos 
    visualize_sampling()
    
    # 2. Benchmark volumétrico de puntos exigidos por el plan de sampling:
    N_int = 5000
    N_b_side = 200
    N_initial = 3000
    
    x_in, _, _ = sample_interior(N=N_int)
    bc_dict = sample_boundary(N_per_side=N_b_side)
    x_ic, _, _ = sample_initial(N=N_initial)
    
    print("\n--- CONTEO VECTORIAL GENERADO ---")
    print(f"  • Puntos Interiores (Collocation):   {x_in.shape[0]}")
    print(f"  • Puntos de Borde y PEC (4 Sides):   {N_b_side * 4}")
    print(f"  • Puntos Instante Inicial t=0:       {x_ic.shape[0]}")
    print("---------------------------------")
