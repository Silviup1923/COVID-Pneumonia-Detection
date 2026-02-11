# ═══════════════════════════════════════════════════════════════════════════════
# 🏥 DETECCIÓN DE COVID-19 Y PNEUMONIA EN CHEST X-RAYS - PYTORCH
# ═══════════════════════════════════════════════════════════════════════════════
#
# DATASET: Chest X-Ray Images (Pneumonia) - Kaggle
# OBJETIVO: Clasificar radiografías en Normal, Bacterial Pneumonia, Viral Pneumonia
# FRAMEWORK: PyTorch puro (sin FastAI)
# 
# PASOS EN KAGGLE:
# 1. Crear nuevo notebook
# 2. Add Data: "Chest X-Ray Images (Pneumonia)" by Paul Mooney
# 3. Activar GPU
# 4. Copiar código celda por celda
#
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 1: INSTALACIÓN DE DEPENDENCIAS
# ═══════════════════════════════════════════════════════════════════════════════

# Librerías especializadas (albumentations para augmentations médicas)
!pip install albumentations --quiet --upgrade

print("✅ Dependencias instaladas")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 2: IMPORTACIONES Y CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# ─── PyTorch ───
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

# ─── Image Processing ───
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

# ─── Data Science ───
import numpy as np
import pandas as pd

# ─── Visualización ───
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline

# ─── Métricas ───
from sklearn.metrics import (
    accuracy_score, confusion_matrix, 
    classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

# ─── Utilidades ───
import os
from pathlib import Path
import glob
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import random

# ─── TensorBoard ───
from torch.utils.tensorboard import SummaryWriter

# ─── Reproducibilidad ───
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("="*70)
print("✅ LIBRERÍAS IMPORTADAS")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🎮 CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"🎲 Semilla: 42")
print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 3: CONFIGURACIÓN DE RUTAS Y PARÁMETROS (CORREGIDA)
# ═══════════════════════════════════════════════════════════════════════════════

from pathlib import Path

# ─── Rutas del Dataset (CORREGIDAS) ───
BASE_PATH = Path('/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray/chest_xray')

# Estructura del dataset:
# chest-xray-pneumonia/
#   ├── train/
#   │   ├── NORMAL/
#   │   └── PNEUMONIA/
#   ├── test/
#   │   ├── NORMAL/
#   │   └── PNEUMONIA/
#   └── val/
#       ├── NORMAL/
#       └── PNEUMONIA/

TRAIN_PATH = BASE_PATH / 'train'
TEST_PATH = BASE_PATH / 'test'
VAL_PATH = BASE_PATH / 'val'

# ─── Verificar que existen ───
print("="*70)
print("🔍 VERIFICACIÓN DE RUTAS")
print("="*70)

print(f"\n📂 BASE_PATH:")
print(f"   {BASE_PATH}")
print(f"   Existe: {'✅' if BASE_PATH.exists() else '❌'}")

print(f"\n📁 TRAIN_PATH:")
print(f"   {TRAIN_PATH}")
print(f"   Existe: {'✅' if TRAIN_PATH.exists() else '❌'}")
if TRAIN_PATH.exists():
    subdirs = [d.name for d in TRAIN_PATH.iterdir() if d.is_dir()]
    print(f"   Subdirectorios: {subdirs}")

print(f"\n📁 TEST_PATH:")
print(f"   {TEST_PATH}")
print(f"   Existe: {'✅' if TEST_PATH.exists() else '❌'}")
if TEST_PATH.exists():
    subdirs = [d.name for d in TEST_PATH.iterdir() if d.is_dir()]
    print(f"   Subdirectorios: {subdirs}")

print(f"\n📁 VAL_PATH:")
print(f"   {VAL_PATH}")
print(f"   Existe: {'✅' if VAL_PATH.exists() else '❌'}")
if VAL_PATH.exists():
    subdirs = [d.name for d in VAL_PATH.iterdir() if d.is_dir()]
    print(f"   Subdirectorios: {subdirs}")

print("\n" + "="*70)

# ─── Rutas de Salida ───
WORK_PATH = Path('/kaggle/working')
MODEL_PATH = WORK_PATH / 'models'
MODEL_PATH.mkdir(exist_ok=True)
LOG_PATH = WORK_PATH / 'logs'
LOG_PATH.mkdir(exist_ok=True)
RESULT_PATH = WORK_PATH / 'results'
RESULT_PATH.mkdir(exist_ok=True)

# ─── Hiperparámetros ───
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
LEARNING_RATE = 1e-4

# ─── Configuración del Problema ───
NUM_CLASSES = 2
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

print("\n⚙️  CONFIGURACIÓN DEL MODELO")
print("="*70)
print(f"Dataset: Chest X-Ray Pneumonia")
print(f"Device: {DEVICE}")
print(f"Classes: {NUM_CLASSES} - {CLASS_NAMES}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 4: EXPLORACIÓN DEL DATASET (ACTUALIZADA - FILTRA MACOSX)
# ═══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("📊 EXPLORACIÓN DEL DATASET")
print("="*70)

def count_images_in_dir(path):
    """
    Cuenta imágenes en un directorio y sus subdirectorios
    Filtra carpetas __MACOSX y archivos ._ (macOS)
    """
    counts = {}
    total = 0
    
    for class_dir in path.iterdir():
        # Filtrar carpetas basura
        if class_dir.is_dir() and class_dir.name != '__MACOSX' and not class_dir.name.startswith('.'):
            # Contar imágenes válidas
            images = []
            for ext in ['*.jpeg', '*.jpg', '*.png']:
                for img_path in class_dir.glob(ext):
                    # Filtrar archivos ._
                    if not img_path.name.startswith('._'):
                        images.append(img_path)
            
            count = len(images)
            counts[class_dir.name] = count
            total += count
    
    return counts, total

# ─── Explorar Train ───
if TRAIN_PATH.exists():
    train_counts, train_total = count_images_in_dir(TRAIN_PATH)
    print(f"\n📁 TRAIN SET:")
    print(f"   Total: {train_total:,} imágenes")
    for class_name, count in sorted(train_counts.items()):
        pct = count/train_total*100 if train_total > 0 else 0
        print(f"   ├─ {class_name}: {count:,} ({pct:.1f}%)")
else:
    print(f"\n❌ TRAIN PATH no encontrado: {TRAIN_PATH}")
    train_total = 0

# ─── Explorar Val ───
if VAL_PATH.exists():
    val_counts, val_total = count_images_in_dir(VAL_PATH)
    print(f"\n📁 VAL SET:")
    print(f"   Total: {val_total:,} imágenes")
    for class_name, count in sorted(val_counts.items()):
        pct = count/val_total*100 if val_total > 0 else 0
        print(f"   ├─ {class_name}: {count:,} ({pct:.1f}%)")
else:
    print(f"\n❌ VAL PATH no encontrado: {VAL_PATH}")
    val_total = 0

# ─── Explorar Test ───
if TEST_PATH.exists():
    test_counts, test_total = count_images_in_dir(TEST_PATH)
    print(f"\n📁 TEST SET:")
    print(f"   Total: {test_total:,} imágenes")
    for class_name, count in sorted(test_counts.items()):
        pct = count/test_total*100 if test_total > 0 else 0
        print(f"   ├─ {class_name}: {count:,} ({pct:.1f}%)")
else:
    print(f"\n❌ TEST PATH no encontrado: {TEST_PATH}")
    test_total = 0

# ─── Resumen Total ───
total_images = train_total + val_total + test_total
if total_images > 0:
    print(f"\n📊 RESUMEN TOTAL:")
    print(f"   Total de imágenes: {total_images:,}")
    print(f"   Train: {train_total:,} ({train_total/total_images*100:.1f}%)")
    print(f"   Val:   {val_total:,} ({val_total/total_images*100:.1f}%)")
    print(f"   Test:  {test_total:,} ({test_total/total_images*100:.1f}%)")
    
    # ─── Visualización ───
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (split_name, counts, total) in enumerate([
        ('TRAIN', train_counts, train_total),
        ('VAL', val_counts, val_total),
        ('TEST', test_counts, test_total)
    ]):
        if total > 0:
            classes = sorted(counts.keys())
            values = [counts[c] for c in classes]
            colors = ['#2ecc71', '#e74c3c']
            
            axes[idx].bar(classes, values, color=colors)
            axes[idx].set_title(f'{split_name}\n({total:,} imgs)', fontweight='bold')
            axes[idx].set_ylabel('Imágenes')
            
            for i, v in enumerate(values):
                pct = v/total*100
                axes[idx].text(i, v + max(values)*0.02, 
                              f'{v:,}\n({pct:.1f}%)', 
                              ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULT_PATH / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "="*70)
print("✅ EXPLORACIÓN COMPLETADA")
print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 5: VISUALIZAR IMÁGENES DE MUESTRA
# ═══════════════════════════════════════════════════════════════════════════════

def show_sample_images(base_path, num_samples=4):
    """Muestra imágenes de muestra de cada clase"""
    
    classes = [d.name for d in base_path.iterdir() if d.is_dir()]
    
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 4*len(classes)))
    
    if len(classes) == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(classes):
        class_path = base_path / class_name
        images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
        
        for j in range(min(num_samples, len(images))):
            img_path = images[j]
            img = Image.open(img_path)
            
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_title(f'{class_name}\n{img_path.name}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULT_PATH / 'sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()

print("🖼️  Muestras de imágenes del TRAIN set:")
show_sample_images(TRAIN_PATH, num_samples=4)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 6: CREAR DATASET CUSTOM DE PYTORCH
# ═══════════════════════════════════════════════════════════════════════════════

class ChestXrayDataset(Dataset):
    """
    Dataset para radiografías de tórax (COVID/Pneumonia)
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: Lista de rutas a imágenes
            labels: Lista de labels (0, 1, 2, etc.)
            transform: Transformaciones a aplicar
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Aplicar transformaciones
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Obtener label
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

print("✅ ChestXrayDataset creado")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 7: PREPARAR LISTAS DE IMÁGENES Y LABELS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data_from_directory(base_path, class_names):
    """
    Carga rutas de imágenes y labels desde estructura de carpetas
    
    Returns:
        image_paths: Lista de Path objects
        labels: Lista de integers (0, 1, 2, etc.)
    """
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_path = base_path / class_name
        
        if not class_path.exists():
            continue
        
        # Buscar todas las imágenes
        images = list(class_path.glob('*.jpeg')) + \
                 list(class_path.glob('*.jpg')) + \
                 list(class_path.glob('*.png'))
        
        for img_path in images:
            image_paths.append(img_path)
            labels.append(class_idx)
    
    return image_paths, labels

# Cargar datos de train
train_paths, train_labels = load_data_from_directory(TRAIN_PATH, CLASS_NAMES)

# Cargar datos de val
val_paths, val_labels = load_data_from_directory(VAL_PATH, CLASS_NAMES)

# Cargar datos de test
test_paths, test_labels = load_data_from_directory(TEST_PATH, CLASS_NAMES)

print("="*70)
print("📊 DATOS CARGADOS")
print("="*70)
print(f"\nTrain: {len(train_paths)} imágenes")
print(f"Val:   {len(val_paths)} imágenes")
print(f"Test:  {len(test_paths)} imágenes")

# Mostrar distribución
for split_name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{split_name} Distribution:")
    for cls, count in zip(unique, counts):
        pct = count/len(labels)*100
        print(f"   Clase {cls} ({CLASS_NAMES[cls]}): {count:,} ({pct:.1f}%)")

print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 8: DEFINIR TRANSFORMACIONES (AUGMENTATIONS)
# ═══════════════════════════════════════════════════════════════════════════════

# Transformaciones para TRAIN (con augmentation)
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Transformaciones para VAL/TEST (sin augmentation)
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

print("✅ Transformaciones definidas")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 9: CREAR DATASETS Y DATALOADERS
# ═══════════════════════════════════════════════════════════════════════════════

# Crear Datasets
train_dataset = ChestXrayDataset(train_paths, train_labels, train_transform)
val_dataset = ChestXrayDataset(val_paths, val_labels, val_transform)
test_dataset = ChestXrayDataset(test_paths, test_labels, val_transform)

# Crear DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("✅ DataLoaders creados")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 10: VERIFICAR UN BATCH
# ═══════════════════════════════════════════════════════════════════════════════

images, labels = next(iter(train_loader))

print(f"📦 Batch shape:")
print(f"   Images: {images.shape}")
print(f"   Labels: {labels.shape}")

# Visualizar batch
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()

for i in range(min(8, len(images))):
    img = images[i].clone()
    img = denormalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    
    axes[i].imshow(img_np)
    axes[i].set_title(f'Label: {CLASS_NAMES[labels[i]]}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(RESULT_PATH / 'batch_sample.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 11: CREAR MODELO (ResNet34)
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(num_classes, pretrained=True):
    model = models.resnet34(weights='DEFAULT' if pretrained else None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

model = create_model(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)

print("✅ Modelo ResNet34 creado")
print(f"   Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 12: LOSS, OPTIMIZER Y SCHEDULER CON CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

# Calcular class weights para dataset desbalanceado
unique_train_labels, counts_train_labels = np.unique(train_labels, return_counts=True)
class_weights = torch.tensor([
    len(train_labels) / (len(unique_train_labels) * count)
    for count in counts_train_labels
], dtype=torch.float32).to(DEVICE)

print("⚖️  Class Weights:")
for cls, weight in enumerate(class_weights):
    print(f"   Clase {cls} ({CLASS_NAMES[cls]}): {weight:.4f}")

# Loss con weights
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7
)

print("✅ Loss, Optimizer y Scheduler configurados")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 13: TENSORBOARD
# ═══════════════════════════════════════════════════════════════════════════════

log_dir = LOG_PATH / f'chest_xray_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
writer = SummaryWriter(log_dir)

print(f"📊 TensorBoard: {log_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 14: FUNCIONES DE TRAIN Y VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS} [TRAIN]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{EPOCHS} [VAL]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels

print("✅ Funciones de entrenamiento definidas")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 15: LOOP DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

best_val_loss = float('inf')
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("="*70)
print("🚀 INICIANDO ENTRENAMIENTO")
print("="*70)

for epoch in range(EPOCHS):
    print(f"\n{'='*70}")
    print(f"ÉPOCA {epoch+1}/{EPOCHS}")
    print(f"{'='*70}")
    
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, epoch
    )
    
    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, DEVICE, epoch
    )
    
    scheduler.step(val_loss)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    print(f"\n📊 RESUMEN ÉPOCA {epoch+1}:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        
        checkpoint_path = MODEL_PATH / 'best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, checkpoint_path)
        
        print(f"   ✅ Mejor modelo guardado")

writer.close()

print("\n" + "="*70)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*70)
print(f"Mejor Val Loss: {best_val_loss:.4f}")
print(f"Mejor Val Acc: {best_val_acc:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 16: VISUALIZAR CURVAS
# ═══════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(val_losses, label='Val Loss', marker='o')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.set_title('Curvas de Pérdida')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Acc', marker='o')
ax2.plot(val_accs, label='Val Acc', marker='o')
ax2.set_xlabel('Época')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Curvas de Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(RESULT_PATH / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 17: MATRIZ DE CONFUSIÓN
# ═══════════════════════════════════════════════════════════════════════════════

cm = confusion_matrix(val_labels, val_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Matriz de Confusión - Validation Set')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig(RESULT_PATH / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 18: MÉTRICAS MÉDICAS
# ═══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("🏥 MÉTRICAS MÉDICAS")
print("="*70)

print(f"\n📋 Classification Report:")
print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES, 
                           labels=list(range(NUM_CLASSES)), digits=3, zero_division=0))

# Si es clasificación binaria, calcular métricas médicas
if NUM_CLASSES == 2 and cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print(f"\n📊 Métricas Clínicas:")
    print(f"   Sensitivity: {sensitivity*100:.2f}% (Recall para clase PNEUMONIA)")
    print(f"   Specificity: {specificity*100:.2f}% (TNR)")
    print(f"   PPV (Precision): {ppv*100:.2f}%")
    print(f"   NPV: {npv*100:.2f}%")

print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 19: EVALUACIÓN EN TEST SET
# ═══════════════════════════════════════════════════════════════════════════════

print("="*70)
print("🧪 EVALUACIÓN EN TEST SET")
print("="*70)

# Cargar mejor modelo
checkpoint = torch.load(MODEL_PATH / 'best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluar
test_loss, test_acc, test_preds, test_labels = validate(
    model, test_loader, criterion, DEVICE, epoch=0
)

print(f"\n📊 RESULTADOS EN TEST SET:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Acc:  {test_acc:.2f}%")

print(f"\n📋 Classification Report (Test):")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES,
                           labels=list(range(NUM_CLASSES)), digits=3, zero_division=0))

# Matriz de confusión test
cm_test = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Matriz de Confusión - Test Set')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig(RESULT_PATH / 'confusion_matrix_test.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# CELDA 20: TENSORBOARD
# ═══════════════════════════════════════════════════════════════════════════════

%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs


# ═══════════════════════════════════════════════════════════════════════════════
# FIN DEL CÓDIGO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("PROYECTO COMPLETADO")
print("="*70)
print("Archivos generados:")
print(f"  - Mejor modelo: {MODEL_PATH / 'best_model.pth'}")
print(f"  - Visualizaciones: {RESULT_PATH}")
print(f"  - TensorBoard logs: {log_dir}")
print("="*70)

