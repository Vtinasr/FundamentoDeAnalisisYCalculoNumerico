import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Configurar matplotlib para evitar múltiples ventanas
plt.close('all')  # Cierra todas las figuras existentes al inicio
plt.ion()  # Modo interactivo (opcional, pero útil para evitar bloqueos)

# 1. Cargar el dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 2. Función para mostrar imágenes (solo una ventana)
def imshow(image):
    """Muestra una imagen de Fashion-MNIST en una sola ventana"""
    plt.close('all')  # Cierra cualquier figura previa
    plt.figure()
    image = image.numpy().squeeze()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show(block=False)  # No bloquea la ejecución
    plt.pause(2)  # Pausa para visualización (ajustable)

# Mostrar primera imagen del dataset
images, labels = next(iter(trainloader))
imshow(images[0])

# 3. Definir y entrenar la red (igual que antes)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Función para visualizar clasificación (solo una ventana)
def view_classify(img, ps):
    """Visualiza la imagen y predicciones en una sola ventana"""
    plt.close('all')  # Cierra cualquier figura previa
    classes = ['Camiseta/top','Pantalón', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatillas', 'Bolso', 'Botín']
    
    fig = plt.figure(figsize=(6,9))
    
    # Mostrar imagen
    plt.subplot(2,1,1)
    img = img.numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    # Mostrar probabilidades
    plt.subplot(2,1,2)
    ps = ps.data.numpy().squeeze()
    plt.barh(np.arange(10), ps)
    plt.yticks(np.arange(10), classes)
    plt.xlabel('Probabilidad')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)  # Pausa para visualización

# 5. Entrenamiento (1 época)
for e in range(1):
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# 6. Probar con una imagen de test
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
images, labels = next(iter(testloader))
img = images[0].view(1, 784)

with torch.no_grad():
    logits = model(img)
ps = F.softmax(logits, dim=1)

# Visualizar resultados (solo una ventana)
view_classify(images[0], ps)

# Opcional: Mantener la última ventana abierta hasta que el usuario la cierre
plt.show(block=True)  # Solo para la última gráfica