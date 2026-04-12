import os
import sys
import argparse
import torch
import numpy as np
import time

import train
import metrics
import visualization
import higher_mode
import sensitivity
from model import FCN

def set_seed(seed=42):
    """Fija la pseudo-aleatoriedad para garantizar que los resultados sean reproducibles matemáticamente."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Conmutador inteligente que empuja los grafos al GPU si NVIDIA/ROCm está disponible."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[!] Hardware detectado: Utilizando motor acelerador en {device.upper()}.")
    return device

def run_training(device, args):
    """Ejecuta el protocolo principal de hiper-entrenamiento Híbrido."""
    print("\n" + "="*60)
    print(" [>] INICIANDO PROTOCOLO DE ENTRENAMIENTO PRINCIPAL TM11")
    print("="*60)
    
    # Arquitectura Base
    model = FCN([3, 128, 128, 128, 128, 3])
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/maxwell_pinn.pth'
    
    start_t = time.time()
    
    epochs_a = 50 if args.quick else 10000
    epochs_b = 10 if args.quick else 1000
    
    model, history = train.train_full(
        model, 
        adam_epochs=epochs_a, 
        lbfgs_iter=epochs_b, 
        device=device, 
        save_path=save_path
    )
    
    end_t = time.time()
    print(f"\n[+] Red Neuronal Entrenada Exitosamente en {(end_t - start_t)/60:.2f} Minutos.")
    print(f"Pesos de conectividad asegurados localmente en {save_path}")
    
    if args.plot_history:
        print("Generando Gráficos Termodinámicos de Conclusión...")
        visualization.plot_training_history(history)
        print("[+] Log de época guardado en 'results/training_history.png'")
        
    return model

def run_evaluation(device, model=None):
    """Carga los pesos (si no están en memoria) y extrae las métricas matemáticas objetivas."""
    print("\n" + "="*60)
    print(" [~] EXTRAYENDO MÉTRICAS Y REPORTES TENSOR-GEOMÉTRICOS")
    print("="*60)
    
    if model is None:
        save_path = 'results/maxwell_pinn.pth'
        if not os.path.exists(save_path):
            print("[-] Error Crítico: No existe un modelo previo entrenado. Corre con --train primero.")
            sys.exit(1)
        model = train.load_checkpoint(path=save_path, layers=[3, 128, 128, 128, 128, 3], device=device)
        
    metrics.full_report(model)
    
    print("\n[  Renderizando Módulo Gráfico y Mapas Ópticos en Carpeta /results  ]")
    print('  - Renderizando Fotografía del Campo 2D en plano inicial...')
    visualization.plot_fields_snapshot(model, t_val=0.0)
    
    print('  - Renderizando Comportamiento Espectral a lo largo de T (Sensor Probe)...')
    visualization.plot_time_evolution(model)
    
    print('  - Generando Comprobación del Teorema Teórico de Poynting (Conservación T/2)...')
    visualization.plot_electromagnetic_energy(model)
    
    print('  - Generando Ploteo Múltiple Vectorial y Curvando Streamlines Magnéticos...')
    visualization.plot_field_lines(model, t_val=0.01)

def main():
    parser = argparse.ArgumentParser(description="""
        P.I.N.N. Solver ⚡ - Simulador Cuántico Electromagnético
        Resuelve las Ecuaciones de Maxwell (Modo TM_11 Rectangular PEC)
    """, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--train', action='store_true', help="Empieza un entrenamiento Adam/LBFGS de 0.")
    parser.add_argument('--evaluate', action='store_true', help="Evalúa las métricas de un .pth existente.")
    parser.add_argument('--full-pipeline', action='store_true', help="Pase OMNI: Entrena, Mide, Documenta y Grafica de una vez.")
    parser.add_argument('--higher-mode', action='store_true', help="Corre el test experimental en TM21 contra TM11.")
    parser.add_argument('--sensitivity', action='store_true', help="Dispara el analizador de hiper-sensibilidad paramétrica.")
    parser.add_argument('--quick', action='store_true', help="Bandera Dummy: Fuerza epocas=50 en todos lados para validar test.")
    parser.add_argument('--plot-history', action='store_true', help="Acompañado de Train, solicita gráfica de los epochs.")

    args = parser.parse_args()
    
    # Semilla de validación algorítmica constante
    set_seed(42)
    device = get_device()
    
    # ----------------------------------------------------
    # ROUTER INTELIGENTE DE MODOS
    # ----------------------------------------------------
    if args.full_pipeline:
        model = run_training(device, args)
        run_evaluation(device, model=model)
        
    elif args.train:
        run_training(device, args)
        
    elif args.evaluate:
        run_evaluation(device)
        
    elif args.higher_mode:
        print("\n[!] LIGANDO MÓDULO EXPERIMENTAL TOPOLÓGICO: TM21")
        epochs_a = 50 if args.quick else 15000
        epochs_b = 10 if args.quick else 1000
        mod_21, _ = higher_mode.train_higher_mode(m=2, n=1, adam_epochs=epochs_a, lbfgs_iter=epochs_b, device=device)
        
        # Test baseline load si existe, sino fake para comparison
        if os.path.exists('results/maxwell_pinn.pth'):
            mod_11 = train.load_checkpoint(device=device)
            higher_mode.compare_predictions(mod_11, mod_21, t_val=0.0)
            print("[+] Gráficas Paralelas de ambos espectros guardadas.")
        else:
            print("[-] No se encontró TM11 base entrenado. Faltan gráficas comparativas. Entrena el pth base primero.")
            
    elif args.sensitivity:
        print("\n[!] ORQUESTANDO RECOLECCIÓN PANDAS PARA MÉTRICAS DE HIPERPARÁMETROS")
        # Simula agregar --full manualmente para el script si no es --quick
        # Para integrarlo lo vamos a invocar mediante librerías nativas usando os.system temporal o parámetros
        if args.quick:
            sys.argv = ['sensitivity.py']
        else:
            sys.argv = ['sensitivity.py', '--full']
            
        print("Traspasando pipeline al ejecutor de sensibilidad interno...")
        # Llama a las funciones principales directamente
        import sensitivity 
        # Modificamos los sub-epochs asertivamente para ahorrar memoria
        eps = 20 if args.quick else 5000
        
        import pandas as pd
        
        ar_cfg = [[3, 32, 32, 3] if args.quick else [3, 128, 128, 128, 128, 3]]
        cl_cfg = [500, 1000] if args.quick else [1000, 5000, 10000]
        lm_cfg = [{'lambda_pde': 1.0, 'lambda_bc': 1.0, 'lambda_ic': 1.0}, {'lambda_pde': 1.0, 'lambda_bc': 10.0, 'lambda_ic': 10.0}]
        
        da = sensitivity.sensitivity_architecture(ar_cfg, adam_epochs=eps, device=device)
        dc = sensitivity.sensitivity_collocation(cl_cfg, adam_epochs=eps, device=device)
        dl = sensitivity.sensitivity_lambdas(lm_cfg, adam_epochs=eps, device=device)
        
        sensitivity.print_sensitivity_tables(da, dc, dl)
        sensitivity.plot_sensitivity_results(da, dc, dl)
        print("[+] Reporte de Sensibilidad concluido.")
    else:
        # Default o sin argumentos
        parser.print_help()
        print("\nPara ejecutar de inmediato de principio a fin usa: python main.py --full-pipeline")

if __name__ == '__main__':
    main()
