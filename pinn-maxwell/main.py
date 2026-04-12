import os
import torch
import numpy as np

from model import MaxwellPINN
from train import train_model
from visualization import plot_snapshot
from metrics import compute_l2_error
from analytical import exact_solution, L, T_res

def main():
    # 1. Configurar directorios
    out_dir = "results"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. Inicializar el modelo
    print("Inicializando arquitectura de red neuronal PINN...")
    model = MaxwellPINN()
    
    # 3. Entrenar el modelo
    # Parámetros por default dados internamente
    print("Comenzando el proceso de entrenamiento de la Física...")
    model = train_model(model, epochs_adam=1500, epochs_lbfgs=500, N_colloc=10000, N_ic=2500, N_bc=2500)
    
    # 4. Guardar pesos del modelo
    weight_path = os.path.join(out_dir, "pinn_maxwell_model.pt")
    torch.save(model.state_dict(), weight_path)
    print(f"Modelo exitosamente guardado en {weight_path}")
    
    # 5. Evaluar precisión final (Métricas Test)
    print("Midiendo errores relativos en el dominio de la evaluación...")
    
    # Generar muestras lineales randomizadas uniformes para evaluación
    N_test = 10000
    x_test = torch.tensor(np.random.uniform(0, L, N_test).astype(np.float32)).unsqueeze(1)
    y_test = torch.tensor(np.random.uniform(0, L, N_test).astype(np.float32)).unsqueeze(1)
    T_max = 2 * T_res
    t_test = torch.tensor(np.random.uniform(0, T_max, N_test).astype(np.float32)).unsqueeze(1)
    
    Ez_ex, Bx_ex, By_ex = exact_solution(x_test, y_test, t_test)
    preds = model(torch.cat([x_test, y_test, t_test], dim=1))
    Ez_pred, Bx_pred, By_pred = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]
    
    err_Ez = compute_l2_error(Ez_pred, Ez_ex)
    err_Bx = compute_l2_error(Bx_pred, Bx_ex)
    err_By = compute_l2_error(By_pred, By_ex)
    
    print("---- RENDIMIENTO PINN ----")
    print(f"Error L2 Relativo Ez: {err_Ez*100:.3f}%")
    print(f"Error L2 Relativo Bx: {err_Bx*100:.3f}%")
    print(f"Error L2 Relativo By: {err_By*100:.3f}%")
    
    # 6. Generar gráficas (snapshots) a t=0.25 T_res y t=0.5 T_res
    print("Creando snapshots e instantaneas...")
    plot_snapshot(model, t_snapshot=T_res*0.25, save_path=os.path.join(out_dir, "snapshot_t025.png"))
    plot_snapshot(model, t_snapshot=T_res*0.50, save_path=os.path.join(out_dir, "snapshot_t050.png"))
    plot_snapshot(model, t_snapshot=T_res*1.00, save_path=os.path.join(out_dir, "snapshot_t100.png"))
    
    print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
