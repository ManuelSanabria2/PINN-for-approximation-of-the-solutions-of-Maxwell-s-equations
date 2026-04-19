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

# ─── Configuración Global de Arquitectura ─────────────────────────────────────
# Centralizar aquí garantiza que train, evaluate, y load_checkpoint
# usen siempre la misma arquitectura sin riesgo de desincronización.
ARCH_LAYERS     = [3, 128, 128, 128, 128, 3]
FOURIER_ON      = True    # Fourier Feature Encoding habilitado
N_FOURIER       = 32      # Frecuencias en el banco de filtros
SIGMA_FOURIER   = 1.5     # omega_TM11/π = π√2/π ≈ 1.41 — ajustado a las frecuencias reales del modo
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed=42):
    """Fija la pseudo-aleatoriedad para garantizar que los resultados sean reproducibles matemáticamente.
    
    IMPORTANTE con Fourier Features: la matriz B es generada con torch.randn dentro de FCN.__init__.
    Llamar set_seed() ANTES de instanciar FCN garantiza que B sea idéntica entre runs,
    haciendo que load_checkpoint() restaure exactamente el mismo banco de frecuencias.
    """
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

    # Arquitectura con Fourier Feature Encoding
    model = FCN(ARCH_LAYERS,
                fourier_features=FOURIER_ON,
                n_fourier=N_FOURIER,
                sigma=SIGMA_FOURIER)

    n_params = model.count_parameters()
    print(f"[!] Red inicializada: FCN{ARCH_LAYERS}")
    print(f"    Fourier Encoding: {'ON' if FOURIER_ON else 'OFF'} "
          f"(n_fourier={N_FOURIER}, sigma={SIGMA_FOURIER})")
    print(f"    Parámetros entrenables: {n_params:,}")

    os.makedirs('results', exist_ok=True)
    save_path = 'results/maxwell_pinn.pth'

    start_t = time.time()

    # Épocas ajustadas: 30k Adam + 5k L-BFGS en modo completo
    epochs_a = 50  if args.quick else 30000
    epochs_b = 10  if args.quick else 5000

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
        print("Generando Gráficos de Convergencia...")
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
        # Cargar con los mismos parámetros de arquitectura usados en entrenamiento
        model = train.load_checkpoint(
            path=save_path,
            layers=ARCH_LAYERS,
            device=device,
            fourier_features=FOURIER_ON,
            n_fourier=N_FOURIER,
            sigma=SIGMA_FOURIER
        )

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
    parser = argparse.ArgumentParser(description="""\
        P.I.N.N. Solver ⚡ - Simulador Cuántico Electromagnético
        Resuelve las Ecuaciones de Maxwell (Modo TM_11 Rectangular PEC)
        Arquitectura: FCN con Fourier Feature Encoding + Adam→L-BFGS
    """, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--train',         action='store_true', help="Empieza un entrenamiento Adam/LBFGS de 0.")
    parser.add_argument('--evaluate',      action='store_true', help="Evalúa las métricas de un .pth existente.")
    parser.add_argument('--full-pipeline', action='store_true', help="Pase OMNI: Entrena, Mide, Documenta y Grafica de una vez.")
    parser.add_argument('--higher-mode',   action='store_true', help="Corre el test experimental en TM21 contra TM11.")
    parser.add_argument('--sensitivity',   action='store_true', help="Dispara el analizador de hiper-sensibilidad paramétrica.")
    parser.add_argument('--quick',         action='store_true', help="Bandera Dummy: Fuerza epochs=50 para validar el pipeline.")
    parser.add_argument('--plot-history',  action='store_true', help="Acompañado de Train, solicita gráfica de los epochs.")

    args = parser.parse_args()

    # Semilla constante ANTES de crear cualquier modelo
    # (garantiza reproducibilidad de la matriz B del Fourier encoding)
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
        epochs_a = 50    if args.quick else 20000
        epochs_b = 10    if args.quick else 2000
        mod_21, _ = higher_mode.train_higher_mode(
            m=2, n=1, adam_epochs=epochs_a, lbfgs_iter=epochs_b, device=device
        )

        # Test baseline load si existe, sino skip comparativa
        if os.path.exists('results/maxwell_pinn.pth'):
            mod_11 = train.load_checkpoint(
                device=device,
                fourier_features=FOURIER_ON,
                n_fourier=N_FOURIER,
                sigma=SIGMA_FOURIER
            )
            higher_mode.compare_predictions(mod_11, mod_21, t_val=0.0)
            print("[+] Gráficas Paralelas de ambos espectros guardadas.")
        else:
            print("[-] No se encontró TM11 base entrenado. Faltan gráficas comparativas.")

    elif args.sensitivity:
        print("\n[!] ORQUESTANDO RECOLECCIÓN PANDAS PARA MÉTRICAS DE HIPERPARÁMETROS")
        import pandas as pd

        eps = 20 if args.quick else 5000

        ar_cfg = [[3, 32, 32, 3]] if args.quick else [
            [3, 64, 64, 64, 3],
            [3, 128, 128, 128, 128, 3]
        ]
        cl_cfg = [500, 1000] if args.quick else [1000, 5000, 10000]
        lm_cfg = [
            {'lambda_pde': 1.0, 'lambda_bc': 1.0,  'lambda_ic': 1.0},
            {'lambda_pde': 1.0, 'lambda_bc': 10.0, 'lambda_ic': 10.0}
        ]

        da = sensitivity.sensitivity_architecture(ar_cfg, adam_epochs=eps, device=device)
        dc = sensitivity.sensitivity_collocation(cl_cfg, adam_epochs=eps, device=device)
        dl = sensitivity.sensitivity_lambdas(lm_cfg, adam_epochs=eps, device=device)

        sensitivity.print_sensitivity_tables(da, dc, dl)
        sensitivity.plot_sensitivity_results(da, dc, dl)
        print("[+] Reporte de Sensibilidad concluido.")

    else:
        # Default o sin argumentos
        parser.print_help()
        print("\nPara ejecutar de principio a fin usa: python main.py --full-pipeline")
        print("Para entrenar solo:                   python main.py --train --plot-history")
        print("Para evaluar un .pth existente:       python main.py --evaluate")


if __name__ == '__main__':
    main()
