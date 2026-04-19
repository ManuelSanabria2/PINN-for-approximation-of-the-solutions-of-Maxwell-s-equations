import math
import torch
import torch.nn as nn

class FCN(nn.Module):
    """
    Fully Connected Network (FCN) de propósito general optimizada para PINNs.
    
    Esta red mapea un vector de entrada de dimensión arbitraria a un vector
    de salida de dimensión arbitraria, actuando como una aproximadora
    universal de funciones vectoriales. No contiene ninguna referencia algorítmica
    o matemática a un sistema físico específico.
    
    Detalles de diseño importantes para Physics-Informed Neural Networks:
    
    1. Activación Tanh vs ReLU:
       Se prefiere Tanh porque es de clase infinita (C^∞), permitiendo
       diferenciación continua de cualquier orden (primera, segunda derivada, etc.)
       necesarias para ecuaciones diferenciales de orden superior. Funciones
       como ReLU tienen segundas derivadas nulas, rompiendo los gradientes por la
       naturaleza constante por secciones de sus primera derivada, inutilizando 
       el error propagado de la PDE.
       
    2. Retorno en Lista Segmentada:
       La red devuelve una lista de tensores individuales de forma (N, 1),
       en lugar de un único tensor multivariado final de la forma (N, M_outputs).
       Extraer los tensores pre-segmentados asila el grafo del autograd 
       por cada variable dependiente individual, facilitando los cálculos sin
       fricciones de las derivadas parciales individuales (por ejemplo calculando
       ∂Ez/∂t con la primera salida de forma aislada).
    
    3. Fourier Feature Encoding (Opcional):
       Al activar fourier_features=True, la capa de entrada proyecta (x, y, t)
       a representaciones sinusoidales usando una matriz aleatoria congelada
       B ∈ R^{n_inputs × n_fourier} escalada por sigma.
       
       φ(v) = [sin(π · B^T · v), cos(π · B^T · v)]  ∈ R^{2·n_fourier}
       
       Esto combate el "Spectral Bias" intrínseco de las FCN: la tendencia de
       las redes con activaciones suaves a aprender funciones de baja frecuencia
       primero, ignorando los armónicos de alta frecuencia presentes en los modos
       de Maxwell (sin(kx·x)·sin(ky·y)·cos(ωt)).
       
       La matriz B es un buffer congelado (no se incluye en optimizer.parameters()),
       actuando como un banco de filtros fijos de frecuencia que alimentan al MLP.
       
       El autograd de PyTorch propaga gradientes correctamente a través del encoding
       porque opera sobre los inputs (x, y, t) que tienen requires_grad=True.
    """

    def __init__(self, layers, fourier_features=False, n_fourier=32, sigma=1.0):
        """
        Args:
            layers (list[int]): Dimensiones de la red, ej. [3, 128, 128, 128, 128, 3].
                                 layers[0] = dimensión de entrada (3 para x, y, t).
                                 layers[-1] = dimensión de salida (3 para Ez, Bx, By).
            fourier_features (bool): Si True, aplica Fourier Feature Encoding a la entrada.
                                      Recomendado para problemas con soluciones oscilatorias.
            n_fourier (int): Número de frecuencias aleatorias a proyectar. La capa de
                              entrada del MLP recibirá 2·n_fourier características.
            sigma (float): Desviación estándar de la distribución de la matriz B.
                           Controla el ancho de banda del banco de frecuencias.
                           sigma ~= frecuencia de la solución / π es un buen punto de inicio.
        """
        super(FCN, self).__init__()
        self.fourier_features = fourier_features
        self.layers_config = layers

        if fourier_features:
            n_inputs = layers[0]
            # Matriz de proyección aleatoria — congelada durante todo el entrenamiento
            B = torch.randn(n_inputs, n_fourier) * sigma
            self.register_buffer('B', B)
            # La primera capa del MLP recibe 2·n_fourier en lugar de n_inputs
            actual_layers = [2 * n_fourier] + layers[1:]
        else:
            actual_layers = layers

        self.linears = nn.ModuleList()

        # Construir capas lineales basado en los enteros de entrada
        for i in range(len(actual_layers) - 1):
            self.linears.append(nn.Linear(actual_layers[i], actual_layers[i + 1]))

        # Inicialización de Xavier y biases a 0
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_fourier(self, x):
        """
        Aplica Fourier Feature Mapping: R^{n_inputs} → R^{2·n_fourier}.
        
        La proyección B está congelada (buffer), el autograd fluye a través
        de las operaciones sin/cos directamente sobre los inputs del dominio.
        
        Args:
            x (Tensor): shape (N, n_inputs), con requires_grad=True en los inputs originales.
        Returns:
            Tensor: shape (N, 2·n_fourier).
        """
        proj = x @ self.B                                              # (N, n_fourier)
        return torch.cat([torch.sin(math.pi * proj),
                          torch.cos(math.pi * proj)], dim=1)          # (N, 2·n_fourier)

    def forward(self, *inputs):
        # Concatena tensores individuales (N, 1) → (N, n_inputs)
        x = torch.cat(inputs, dim=1)

        # Aplicar Fourier Feature Encoding si está habilitado
        if self.fourier_features:
            x = self.encode_fourier(x)

        # Pasa por capas ocultas empleando tanh
        for i in range(len(self.linears) - 1):
            x = torch.tanh(self.linears[i](x))

        # Capa de salida es lineal
        out = self.linears[-1](x)

        # Regresar outputs individuales en una lista
        # Descomponemos en subtensores de tamaño (N, 1) iterando a través de las columnas de out
        return [out[:, i:i+1] for i in range(out.shape[1])]

    def count_parameters(self):
        """
        Retorna el número total de parámetros entrenables en la red.
        Nota: Si fourier_features=True, la matriz B NO se cuenta (es un buffer congelado).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 55)
    print("  TEST DE ARQUITECTURA FCN — MODO ESTÁNDAR Y FOURIER")
    print("=" * 55)

    # Prueba 1: FCN estándar (backward compatible — sin cambios de interfaz)
    model_std = FCN([3, 128, 128, 128, 128, 3])
    print(f"\n[FCN Estándar]")
    print(f"  Parámetros entrenables: {model_std.count_parameters():,}")

    # Prueba 2: FCN con Fourier Feature Encoding
    model_ff = FCN([3, 128, 128, 128, 128, 3], fourier_features=True, n_fourier=32, sigma=1.0)
    print(f"\n[FCN + Fourier Features (n_fourier=32, sigma=1.0)]")
    print(f"  Parámetros entrenables: {model_ff.count_parameters():,}")
    print(f"  Buffer B (congelado):   {model_ff.B.shape}  →  salida: {2*32}D")

    # Verificar que autograd funciona correctamente con Fourier encoding
    x = torch.rand(10, 1, requires_grad=True)
    y = torch.rand(10, 1, requires_grad=True)
    t = torch.rand(10, 1, requires_grad=True)

    preds = model_ff(x, y, t)

    grad_x = torch.autograd.grad(
        preds[0], x,
        grad_outputs=torch.ones_like(preds[0]),
        create_graph=True
    )[0]
    print(f"\n  ∂Ez/∂x shape: {grad_x.shape}  — Autograd a través de Fourier Encoding: OK ✓")

    grad_xx = torch.autograd.grad(
        grad_x, x,
        grad_outputs=torch.ones_like(grad_x),
        create_graph=False
    )[0]
    print(f"  ∂²Ez/∂x² shape: {grad_xx.shape}  — Derivada de 2do orden: OK ✓")
