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
    """
    def __init__(self, layers):
        super(FCN, self).__init__()
        self.layers = layers
        self.linears = nn.ModuleList()
        
        # Construir capas lineales basado en los enteros de entrada
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
            
        # Inicialización de Xavier y biases a 0
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, *inputs):
        # Concatena tensores individuales (N, 1)
        x = torch.cat(inputs, dim=1)
        
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
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Inicializar prueba de modelo
    model = FCN([3, 128, 128, 128, 128, 3])
    print("Arquitectura FCN inicializada correctamente:")
    print(model)
    print(f"Total de parámetros entrenables: {model.count_parameters()}")
