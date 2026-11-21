# 🩻 CheXpert Deep Learning Preprocessing Pipeline

Pipeline de preprocesamiento y preparación de datos para el dataset **CheXpert**, diseñado para tareas de clasificación multietiqueta de patologías torácicas utilizando **PyTorch** y técnicas de **Deep Learning**.

## 📌 Características Principales

- Filtrado exclusivo de radiografías frontales (AP/PA).
- Extracción y agrupación por paciente para evitar *data leakage*.
- Tratamiento de valores faltantes e incertidumbre clínica (*U-labels*).
- Validación cruzada mediante **GroupKFold**.
- Mejora de contraste con **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- Conservación de la relación de aspecto mediante redimensionamiento con *padding*.
- Aumento de datos clínicamente válido.
- Normalización compatible con modelos preentrenados en ImageNet.
- Implementación de un `Dataset` personalizado en PyTorch.

---

## 📂 Dataset

El proyecto utiliza el conjunto de datos **CheXpert**, una base de datos de radiografías de tórax desarrollada por la Universidad de Stanford para investigación en diagnóstico asistido por inteligencia artificial.

### Descarga

```bash
kaggle datasets download -d ashery/chexpert --unzip
```

---

## ⚙️ Configuración

Los parámetros principales del pipeline se encuentran definidos en el diccionario `CONFIG`.

```python
CONFIG = {
    "seed": 42,
    "n_splits": 5,
    "image_size": 384,
    "batch_size": 32,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": (8, 8),
    "u_smoothing_value": 0.55
}
```

### Parámetros utilizados

| Parámetro | Valor |
|-----------|--------|
| Seed | 42 |
| Folds | 5 |
| Tamaño imagen | 384×384 |
| Batch Size | 32 |
| CLAHE Clip Limit | 2.0 |
| CLAHE Tile Grid | 8×8 |
| U-Smoothing | 0.55 |

---

## 📊 Exploración y Filtrado de Datos

A partir del archivo `train.csv` se realizó un filtrado para conservar únicamente radiografías frontales.

### Estadísticas obtenidas

| Métrica | Valor |
|----------|---------|
| Total registros originales | 223,414 |
| Radiografías frontales | 191,027 |
| Radiografías laterales | 32,387 |
| Pacientes únicos | 64,534 |

Además, se generó una visualización de prevalencia para las 14 patologías incluidas en el estudio, mostrando tanto etiquetas positivas como inciertas.

---

## 🏷️ Tratamiento de Etiquetas

CheXpert utiliza cuatro posibles estados para cada patología:

| Valor | Significado |
|---------|------------|
| 1 | Positivo |
| 0 | Negativo |
| -1 | Incierto |
| NaN | Sin observación |

### Estrategia aplicada

#### Valores faltantes

```python
NaN -> 0.0
```

#### U-Smoothing

Para patologías donde la incertidumbre suele contener información clínica relevante:

- Atelectasis
- Consolidation
- Edema

se aplicó:

```python
-1 -> 0.55
```

Para el resto de patologías:

```python
-1 -> 0.0
```

Esta estrategia permite representar la incertidumbre como una probabilidad intermedia en lugar de descartarla completamente.

---

## 🔀 Validación Cruzada

Para evitar que imágenes pertenecientes al mismo paciente aparezcan simultáneamente en entrenamiento y validación, se implementó:

```python
GroupKFold(n_splits=5)
```

utilizando `patient_id` como grupo de separación.

### Auditoría de particiones

| Fold | Entrenamiento | Validación | Solapamiento |
|--------|--------------|-------------|--------------|
| 0 | 152,821 RX | 38,206 RX | 0 |
| 1 | 152,821 RX | 38,206 RX | 0 |
| 2 | 152,822 RX | 38,205 RX | 0 |
| 3 | 152,822 RX | 38,205 RX | 0 |
| 4 | 152,822 RX | 38,205 RX | 0 |

✅ No se detectó fuga de información entre conjuntos.

---

## 🖼️ Preprocesamiento de Imágenes

### 1. Mejora de Contraste

Cada radiografía es convertida a escala de grises y procesada mediante CLAHE:

```python
cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)
```

**Beneficios:**

- Mayor visibilidad de estructuras pulmonares.
- Mejora local del contraste.
- Conservación de detalles anatómicos.

### 2. Redimensionamiento

Las imágenes originales presentan distintas resoluciones.

El pipeline:

1. Escala proporcionalmente.
2. Conserva la relación de aspecto.
3. Añade padding negro cuando es necesario.
4. Produce imágenes cuadradas de:

```text
384 x 384 píxeles
```

Esta estrategia evita deformaciones anatómicas durante el entrenamiento.

---

## 🔄 Data Augmentation

Las transformaciones aplicadas durante el entrenamiento incluyen:

```python
A.Affine(
    scale=(0.95,1.05),
    translate_percent=(-0.05,0.05),
    rotate=(-7,7)
)
```

```python
A.RandomBrightnessContrast(
    brightness_limit=0.10,
    contrast_limit=0.10
)
```

### Consideraciones Clínicas

No se utiliza:

```python
HorizontalFlip
```

porque invertir lateralmente una radiografía puede alterar la ubicación anatómica del corazón y otras estructuras relevantes para el diagnóstico.

---

## 📐 Normalización

Las imágenes son normalizadas utilizando las estadísticas de ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Esto facilita la reutilización de modelos preentrenados mediante *transfer learning*.

---

## 🧱 Dataset Personalizado

Se implementó una clase personalizada:

```python
CheXpertDataset
```

encargada de:

- Leer la imagen desde disco.
- Aplicar CLAHE.
- Redimensionar y centrar la imagen.
- Ejecutar las transformaciones de Albumentations.
- Convertir la imagen a tensor.
- Cargar las etiquetas multietiqueta correspondientes.

Cada muestra retorna:

```python
(image_tensor, target_tensor)
```

---

## ✅ Verificación del Pipeline

Se realizó una prueba funcional utilizando un `DataLoader`.

### Resultados

```text
Shape imágenes:
(32, 3, 384, 384)

Tipo:
torch.float32

Rango normalizado:
Min = -2.118
Max = 2.640

NaNs:
0

Shape etiquetas:
(32, 14)
```

Estos resultados confirman que los datos quedan correctamente preparados para el entrenamiento de modelos de Deep Learning.

---

## 🛠️ Tecnologías Utilizadas

- Python 3
- PyTorch
- OpenCV
- Albumentations
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Kaggle API

---

## 🚀 Flujo General del Pipeline

```text
CheXpert Dataset
        │
        ▼
Filtrado de imágenes frontales
        │
        ▼
Extracción de Patient ID
        │
        ▼
Tratamiento de etiquetas
(NaN y U-Smoothing)
        │
        ▼
Partición GroupKFold
        │
        ▼
CLAHE + Padding + Resize
        │
        ▼
Data Augmentation
        │
        ▼
Normalización ImageNet
        │
        ▼
Dataset PyTorch
        │
        ▼
DataLoader
        │
        ▼
Entrenamiento Deep Learning
```

---

## 🚀 Próximos Pasos

- Entrenamiento de modelos CNN (DenseNet121, EfficientNet, ResNet).
- Evaluación mediante ROC-AUC por patología.
- Implementación de Early Stopping.
- Ajuste fino de hiperparámetros.
- Interpretabilidad mediante Grad-CAM.
- Despliegue para inferencia sobre nuevas radiografías.

---
# 🧪 Implementación y Pruebas del Modelo de Deep Learning para CheXpert

Esta sección documenta las pruebas realizadas para validar el pipeline de procesamiento de datos, la arquitectura del modelo y el flujo completo de entrenamiento de una red neuronal convolucional para clasificación multietiqueta de patologías torácicas utilizando el dataset CheXpert. 【1-754e35】

---

## 🎯 Objetivo

Verificar que todos los componentes desarrollados en etapas anteriores funcionen correctamente antes de realizar el entrenamiento completo del modelo:

- Carga de imágenes.
- Transformaciones y preprocesamiento.
- Construcción de batches.
- Definición de la red neuronal.
- Forward Pass.
- Función de pérdida.
- Ciclo de entrenamiento.
- Evaluación inicial mediante métricas de clasificación. 【1-754e35】

---

## ✅ Prueba 1: Validación del DataLoader

La primera prueba consistió en comprobar que las imágenes y etiquetas fueran cargadas correctamente desde el `DataLoader`.

### Validaciones realizadas

```python
assert batch_x.shape[1] == 3
assert batch_x.shape[2] == 384
assert batch_x.shape[3] == 384

assert not torch.isnan(batch_x).any()
assert not torch.isnan(batch_y).any()
```

### Verificaciones

- Las imágenes contienen 3 canales RGB.
- Todas las imágenes tienen dimensiones de 384×384 píxeles.
- No existen valores NaN en imágenes.
- No existen valores NaN en etiquetas.
- El pipeline genera lotes listos para entrenamiento. 【1-754e35】

### Resultado esperado

```text
✓ Imágenes correctamente transformadas
✓ Etiquetas correctamente cargadas
✓ Sin valores NaN
```

---

## 🖼️ Prueba 2: Visualización de Imágenes Procesadas

Para verificar visualmente la calidad del preprocesamiento se mostró una muestra de imágenes provenientes del `DataLoader`. 【1-754e35】

![EMBEDDEDIMAGE](placeholder-0)

Antes de la visualización se aplicó la desnormalización utilizando las estadísticas de ImageNet.

```python
img = (
    img * np.array(CONFIG["imagenet_std"])
    + np.array(CONFIG["imagenet_mean"])
)
```

### Objetivo

Confirmar que:

- Las imágenes conservan la información anatómica.
- No existen distorsiones visuales.
- El contraste generado por CLAHE es adecuado.
- Las imágenes mantienen el formato esperado por la red neuronal. 【1-754e35】

---

## 🧠 Prueba 3: Construcción del Modelo

Se implementó una arquitectura basada en **ResNet50 preentrenada** utilizando los pesos por defecto de PyTorch.

```python
model = models.resnet50(weights="DEFAULT")
```

Posteriormente se reemplazó la capa completamente conectada para adaptarla al problema de clasificación multietiqueta de 14 patologías.

```python
model.fc = nn.Linear(
    model.fc.in_features,
    len(CONFIG["target_columns"])
)
```

### Ventajas de ResNet50

- Arquitectura profunda con conexiones residuales.
- Reducción del problema de gradientes desaparecidos.
- Excelente desempeño en tareas médicas mediante transferencia de aprendizaje.
- Compatibilidad con imágenes normalizadas mediante estadísticas de ImageNet. 【1-754e35】

---

## 🔄 Prueba 4: Forward Pass

Una vez construido el modelo se realizó una propagación hacia adelante (*Forward Pass*).

```python
outputs = model(batch_x)
```

Se verificó que la dimensión de salida coincidiera con las 14 patologías del dataset.

```python
assert outputs.shape[1] == 14
```

### Resultado esperado

```text
✓ Forward pass exitoso
```

Esta prueba confirma que la arquitectura puede procesar correctamente un lote completo de imágenes y generar predicciones para todas las clases objetivo. 【1-754e35】

---

## 📉 Prueba 5: Función de Pérdida

Debido a que una radiografía puede presentar múltiples patologías simultáneamente, se utilizó:

```python
nn.BCEWithLogitsLoss()
```

### Justificación

Esta función combina:

- Activación Sigmoid.
- Binary Cross Entropy.

y es una de las alternativas más utilizadas para problemas de clasificación multietiqueta en imágenes médicas. 【1-754e35】

### Cálculo

```python
loss = criterion(
    outputs,
    batch_y
)
```

El valor de pérdida inicial sirve como referencia para evaluar el progreso durante el entrenamiento. 【1-754e35】

---

## 🚀 Prueba 6: Ejecución de una Época de Entrenamiento

Se implementó un ciclo completo de entrenamiento utilizando el optimizador:

```python
torch.optim.AdamW
```

con una tasa de aprendizaje inicial:

```python
lr = 1e-4
```

### Flujo de entrenamiento

```python
optimizer.zero_grad()

preds = model(images)

loss = criterion(preds, labels)

loss.backward()

optimizer.step()
```

Durante la iteración se acumula la pérdida de cada batch para calcular posteriormente la pérdida promedio de la época. 【1-754e35】

### Métrica registrada

```python
Loss promedio
```

Una disminución progresiva de esta métrica indica que el modelo está aprendiendo patrones relevantes presentes en las radiografías. 【1-754e35】

---

## 📊 Prueba 7: Evaluación Inicial del Modelo

Finalizada la primera etapa de entrenamiento, se realizó una evaluación preliminar utilizando métricas de clasificación. 【1-754e35】

![EMBEDDEDIMAGE](placeholder-1)

### Conversión de probabilidades

```python
preds = torch.sigmoid(outputs)
preds_binary = (preds > 0.5).int()
```

### Métricas calculadas

#### F1-Score

```python
f1_score(
    batch_y.numpy(),
    preds_binary.numpy(),
    average="macro"
)
```

Mide el equilibrio entre precisión y exhaustividad.

#### Precision

```python
precision_score(
    batch_y.numpy(),
    preds_binary.numpy(),
    average="macro"
)
```

Indica el porcentaje de predicciones positivas correctas.

#### Recall

```python
recall_score(
    batch_y.numpy(),
    preds_binary.numpy(),
    average="macro"
)
```

Mide la capacidad del modelo para detectar casos positivos.

### Salidas Generadas

```text
F1 Score
Precision
Recall
```

Estas métricas proporcionan una primera estimación del comportamiento del modelo antes de realizar el entrenamiento completo y los ajustes de hiperparámetros. 【1-754e35】

---

## 📈 Flujo General de Implementación

```text
DataLoader
      │
      ▼
Validación de datos
      │
      ▼
Visualización de imágenes
      │
      ▼
Modelo ResNet50
      │
      ▼
Forward Pass
      │
      ▼
BCEWithLogitsLoss
      │
      ▼
AdamW Optimizer
      │
      ▼
Entrenamiento
      │
      ▼
Predicciones
      │
      ▼
F1 - Precision - Recall
```

---

## ✅ Conclusiones

Las pruebas realizadas permitieron verificar el correcto funcionamiento de todos los componentes del sistema. El pipeline procesó adecuadamente las imágenes y etiquetas, la arquitectura ResNet50 generó salidas con la dimensión esperada, la función de pérdida pudo calcularse sin errores y se ejecutó satisfactoriamente una época de entrenamiento. Finalmente, se obtuvieron métricas iniciales de evaluación que servirán como línea base para futuras etapas de optimización y entrenamiento del modelo de clasificación multietiqueta sobre el dataset CheXpert. 【1-754e35】
