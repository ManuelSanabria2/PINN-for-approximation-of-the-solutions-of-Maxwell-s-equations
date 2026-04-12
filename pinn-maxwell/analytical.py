import numpy as np
import matplotlib.pyplot as plt
import os

# Constantes Fundamentales (Unidades Naturales Adimensionales para PINNs)
c = 1.0                    # Velocidad de la luz normalizada
mu0 = 1.0                  # Permeabilidad magnética normalizada
eps0 = 1.0                 # Permitividad eléctrica normalizada
L = 1.0                    # Longitud de la cavidad [m]
E0 = 1.0                   # Amplitud del campo eléctrico

def cavity_mode(m, n):
    """
    Calcula los parámetros topológicos del modo de resonancia transversal TM_{mn}.
    
    Física de Cavidades Resonantes PEC:
    Una cavidad resonante PEC (Perfect Electric Conductor) confina campos electromagnéticos,
    forzando que las componentes del campo eléctrico tangenciales a las paredes metálicas sean 
    numéricamente nulas. Esto genera ondas estacionarias que solo pueden armarse si media 
    longitud de onda cabe un número entero de veces en la cavidad. 
    Debido a esta barrera física, la energía solo puede oscilar a frecuencias discretas, 
    creando un espectro de posibles modos (m, n).
    """
    kx = m * np.pi / L
    ky = n * np.pi / L
    k = np.sqrt(kx**2 + ky**2)
    omega = c * k
    T_period = 2 * np.pi / omega
    freq_Hz = omega / (2 * np.pi)
    
    return {
        'kx': kx,
        'ky': ky,
        'k': k,
        'omega': omega,
        'T_period': T_period,
        'freq_Hz': freq_Hz
    }

def Ez_exact(x, y, t, m=1, n=1):
    """
    Solución analítica predictiva del Campo Eléctrico en su componente Z.
    En el rotacional TM 2D el campo oscila sin propagarse y debe caer a cero en los bordes.
    
    Derivación Analítica:
    Se propone un perfil espacial armónico del tipo Estacionario A*sin(kx*x)*sin(ky*y).
    Esta base temporal multiplicada por la oscilación E0 * cos(ωt) asegura matemáticamente 
    la ecuación de onda.
    """
    params = cavity_mode(m, n)
    kx, ky, omega = params['kx'], params['ky'], params['omega']
    
    return E0 * np.sin(kx * x) * np.sin(ky * y) * np.cos(omega * t)

def Bx_exact(x, y, t, m=1, n=1):
    """
    Solución analítica teórica para el Campo Magnético Bx.
    
    Derivación Analítica (Mediante Faraday x):
    Por la Ec. de Faraday: ∂Bx/∂t = -∂Ez/∂y
    Dado que ∂Ez/∂y = E0 * ky * sin(kx*x) * cos(ky*y) * cos(ωt),
    podemos integrar Bx respecto a t aislando la fase sin(ωt):
    Bx = -E0 * (ky/ω) * sin(kx*x) * cos(ky*y) * sin(ωt)
    """
    params = cavity_mode(m, n)
    kx, ky, omega = params['kx'], params['ky'], params['omega']
    
    return -(ky / omega) * E0 * np.sin(kx * x) * np.cos(ky * y) * np.sin(omega * t)

def By_exact(x, y, t, m=1, n=1):
    """
    Solución analítica teórica para el Campo Magnético By.
    
    Derivación Analítica (Mediante Faraday y):
    Por la Ec. de Faraday: ∂By/∂t = ∂Ez/∂x
    Dado que ∂Ez/∂x = E0 * kx * cos(kx*x) * sin(ky*y) * cos(ωt),
    integramos respecto a t separando términos:
    By = +E0 * (kx/ω) * cos(kx*x) * sin(ky*y) * sin(ωt)
    """
    params = cavity_mode(m, n)
    kx, ky, omega = params['kx'], params['ky'], params['omega']
    
    return (kx / omega) * E0 * np.cos(kx * x) * np.sin(ky * y) * np.sin(omega * t)

def verify_maxwell(m=1, n=1, N=50, N_t=20):
    """
    Valida y comprueba matemáticamente que la base de ground truths creada satisface
    realmente los PDEs en todos sus puntos por medio de Diferencias Finitas sobre la malla.
    """
    params = cavity_mode(m, n)
    T = params['T_period']
    
    # Grids lineales
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    t_lin = np.linspace(0, T, N_t)
    dx = x_lin[1] - x_lin[0]
    dy = y_lin[1] - y_lin[0]
    dt = t_lin[1] - t_lin[0]
    
    X, Y, Time = np.meshgrid(x_lin, y_lin, t_lin, indexing='ij')
    
    # Evaluación Funcional
    Ez = Ez_exact(X, Y, Time, m, n)
    Bx = Bx_exact(X, Y, Time, m, n)
    By = By_exact(X, Y, Time, m, n)
    
    # Diferencias Finitas Numéricas np.gradient(data, dz, dy, dx) para indexing='ij'
    dEz_dx, dEz_dy, dEz_dt = np.gradient(Ez, dx, dy, dt, edge_order=2)
    dBx_dx, dBx_dy, dBx_dt = np.gradient(Bx, dx, dy, dt, edge_order=2)
    dBy_dx, dBy_dy, dBy_dt = np.gradient(By, dx, dy, dt, edge_order=2)
    
    # 4 Ecuaciones Residuales
    R1 = dBx_dt + dEz_dy
    R2 = dBy_dt - dEz_dx
    R3 = mu0 * eps0 * dEz_dt - dBy_dx + dBx_dy
    R4 = dBx_dx + dBy_dy
    
    print("--- VERIFICACIÓN NUMÉRICA MAXWELL (FINITE DIFFERENCES) ---")
    print(f"Residual R1 (Faraday X):       {np.max(np.abs(R1)):.6e}")
    print(f"Residual R2 (Faraday Y):       {np.max(np.abs(R2)):.6e}")
    print(f"Residual R3 (Ampère-Maxwell):  {np.max(np.abs(R3)):.6e}")
    print(f"Residual R4 (Gauss Magnética): {np.max(np.abs(R4)):.6e}")
    
    # Gráficos de confirmación óptica a T=0 y T=T/4
    # Capturamos el índice de slicing
    t0_idx = 0
    tq_idx = int(N_t * 0.25)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Fila 1: t=0
    c1 = axes[0,0].contourf(X[:,:,t0_idx], Y[:,:,t0_idx], Ez[:,:,t0_idx], 30, cmap='RdBu', vmin=-E0, vmax=E0)
    fig.colorbar(c1, ax=axes[0,0])
    axes[0,0].set_title("Ez a t=0")
    
    c2 = axes[0,1].contourf(X[:,:,t0_idx], Y[:,:,t0_idx], Bx[:,:,t0_idx], 30, cmap='Spectral')
    fig.colorbar(c2, ax=axes[0,1])
    axes[0,1].set_title("Bx a t=0 (Teórico es 0)")
    
    c3 = axes[0,2].contourf(X[:,:,t0_idx], Y[:,:,t0_idx], By[:,:,t0_idx], 30, cmap='Spectral')
    fig.colorbar(c3, ax=axes[0,2])
    axes[0,2].set_title("By a t=0 (Teórico es 0)")

    # Fila 2: t=T/4 (Máxima fluctuación inducida en B)
    # Ez es nulo, los campos B están en cresta
    c4 = axes[1,0].contourf(X[:,:,tq_idx], Y[:,:,tq_idx], Ez[:,:,tq_idx], 30, cmap='RdBu', vmin=-E0, vmax=E0)
    fig.colorbar(c4, ax=axes[1,0])
    axes[1,0].set_title("Ez a t=T/4")
    
    c5 = axes[1,1].contourf(X[:,:,tq_idx], Y[:,:,tq_idx], Bx[:,:,tq_idx], 30, cmap='Spectral')
    fig.colorbar(c5, ax=axes[1,1])
    axes[1,1].set_title("Bx a t=T/4")
    
    c6 = axes[1,2].contourf(X[:,:,tq_idx], Y[:,:,tq_idx], By[:,:,tq_idx], 30, cmap='Spectral')
    fig.colorbar(c6, ax=axes[1,2])
    axes[1,2].set_title("By a t=T/4")
    
    plt.tight_layout()
    out_dir = 'results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, 'analytical_verification.png'), dpi=300)
    print(f"--- Gráficas Salvadas en {os.path.join(out_dir, 'analytical_verification.png')} ---")
    plt.close()

def generate_reference(N=50, N_t=20, m=1, n=1):
    """
    Exportación modular de mallas y topologías funcionales.
    Útil para proveer ground truths masivos como validación para L2 Error PINN.
    """
    params = cavity_mode(m, n)
    T = params['T_period']
    omega = params['omega']
    
    x_lin = np.linspace(0, L, N)
    y_lin = np.linspace(0, L, N)
    t_lin = np.linspace(0, T, N_t)
    
    X, Y, Time = np.meshgrid(x_lin, y_lin, t_lin, indexing='ij')
    
    return {
        'X': X,
        'Y': Y,
        'T': Time,
        'Ez': Ez_exact(X, Y, Time, m, n),
        'Bx': Bx_exact(X, Y, Time, m, n),
        'By': By_exact(X, Y, Time, m, n),
        'omega': omega,
        'T_period': T
    }

if __name__ == '__main__':
    # Información Fundamental TM11
    params_TM11 = cavity_mode(1, 1)
    
    print("\n--- MODO FUNDAMENTAL: TM11 ---")
    print(f"Kx (M=1):        {params_TM11['kx']:.4f}")
    print(f"Ky (N=1):        {params_TM11['ky']:.4f}")
    print(f"Omega:           {params_TM11['omega']:.4e} rad/s")
    print(f"Frecuencia (Hz): {params_TM11['freq_Hz']:.4e} Hz")
    print(f"Periodo T:       {params_TM11['T_period']:.4e} s")
    print("-" * 32)
    
    # Comprobar diferencias finitas y generar png
    verify_maxwell(m=1, n=1)
