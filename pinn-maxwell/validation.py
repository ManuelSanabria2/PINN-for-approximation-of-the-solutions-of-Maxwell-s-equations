import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import FCN
from analytical import Ez_exact, Bx_exact, By_exact, L, cavity_mode
import visualization
import metrics

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def run_analytical_benchmark(model, device, t_val=0.1):
    """Prueba 1: Validación contra Soluciones Analíticas"""
    print("\n[1/4] Ejecutando Validación contra Soluciones Analíticas...")
    visualization.plot_fields_snapshot(model, t_val=t_val, save_path='results/validation/')
    errs = metrics.l2_error_per_field(model)
    print(f"      - Error L2 Relativo Ez: {errs['Ez']:.2f}%")
    print(f"      - Error L2 Relativo Bx: {errs['Bx']:.2f}%")
    print(f"      - Error L2 Relativo By: {errs['By']:.2f}%")
    return errs

def run_physics_benchmarks(model, device, t_val=0.1):
    """Prueba 2: Demostración de las Leyes de Conservación"""
    print("\n[2/4] Ejecutando Demostración de Leyes de Conservación...")
    visualization.plot_poynting_vector(model, t_val=t_val, save_path='results/validation/')
    visualization.plot_spatial_residuals(model, t_val=t_val, save_path='results/validation/')
    visualization.plot_electromagnetic_energy(model, save_path='results/validation/')
    
    ene = metrics.energy_conservation_error(model)
    print(f"      - Error de Conservación de Energía: {ene['conservation_error_pct']:.2f}%")
    
    res = metrics.residual_per_equation(model)
    print(f"      - Residuo Medio Faraday: {res['faraday_x']['mean']:.2e}")
    print(f"      - Residuo Medio Ampere: {res['ampere']['mean']:.2e}")

def run_efficiency_benchmark(model, device, N_points=100000):
    """Prueba 3: Comparativa de Eficiencia (Inferencia vs FDTD Mock)"""
    print(f"\n[3/4] Ejecutando Benchmark de Eficiencia ({N_points} puntos)...")
    
    x = torch.rand(N_points, 1).to(device)
    y = torch.rand(N_points, 1).to(device)
    t = torch.rand(N_points, 1).to(device)
    
    # Warm-up
    for _ in range(10): model(x, y, t)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(x, y, t)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / 100.0
    print(f"      - Tiempo promedio de inferencia (Batch {N_points}): {avg_time*1000:.2f} ms")
    print(f"      - Puntos por segundo: {N_points / avg_time:,.0f} pts/s")
    print("      - Nota: A diferencia de FDTD, este cálculo es continuo y libre de malla (Mesh-Free).")

def run_generalization_test(model, device):
    """Prueba 4: Generalización y Robustez (Puntos no vistos)"""
    print("\n[4/4] Ejecutando Pruebas de Generalización...")
    
    # Evaluar en una resolución mucho mayor (ej. 200x200) para ver si hay artefactos de malla
    print("      - Evaluando estabilidad en alta resolución (200x200)...")
    visualization.plot_fields_snapshot(model, t_val=0.25, N=200, save_path='results/validation/high_res/')
    
    # Evaluar en tiempo T > T_train (si aplica)
    params = cavity_mode(1, 1)
    T_extrapol = 3.0 * params['T_period']
    print(f"      - Evaluando extrapolación temporal en t = {T_extrapol:.2f}s...")
    visualization.plot_fields_snapshot(model, t_val=T_extrapol, save_path='results/validation/extrapol/')

def main(model_path='results/maxwell_pinn.pth'):
    device = get_device()
    print(f"\n=== PROTOCOLO DE VALIDACIÓN RIGUROSA MAXWELL PINN ===")
    print(f"Dispositivo: {device.upper()}")
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    # Cargar modelo (usando parámetros por defecto de main.py)
    # Importante: Si cambiaste la arquitectura en main.py, cámbiala aquí también.
    ARCH = [3, 128, 128, 128, 128, 3]
    model = FCN(ARCH, fourier_features=True, n_fourier=32, sigma=1.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    os.makedirs('results/validation', exist_ok=True)
    
    run_analytical_benchmark(model, device)
    run_physics_benchmarks(model, device)
    run_efficiency_benchmark(model, device)
    run_generalization_test(model, device)
    
    print("\n[+] Protocolo de validación completado. Resultados en 'results/validation/'")

if __name__ == '__main__':
    main()
