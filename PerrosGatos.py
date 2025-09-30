import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
import joblib

# Configuración de parámetros
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Ruta de la carpeta con imágenes
dataset_path = "dataset_extracted/dataset"

def cargar_imagenes(ruta_carpeta, etiqueta, max_imagenes=None):
    """
    Carga imágenes desde una carpeta y las asigna a una etiqueta
    """
    imagenes = []
    etiquetas = []
    
    if not os.path.exists(ruta_carpeta):
        print(f"Error: No se encuentra la carpeta {ruta_carpeta}")
        return imagenes, etiquetas
    
    archivos = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_imagenes:
        archivos = archivos[:max_imagenes]
    
    print(f"Cargando {len(archivos)} imágenes desde {ruta_carpeta}")
    
    for i, archivo in enumerate(archivos):
        try:
            # Cargar imagen
            img_path = os.path.join(ruta_carpeta, archivo)
            img = Image.open(img_path).convert('RGB')
            
            # Redimensionar
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            
            # Convertir a array y normalizar
            img_array = np.array(img) / 255.0
            
            # Aplanar la imagen (convertir de 2D a 1D)
            img_flat = img_array.flatten()
            
            imagenes.append(img_flat)
            etiquetas.append(etiqueta)
            
            if (i + 1) % 100 == 0:
                print(f"Procesadas {i + 1} imágenes...")
                
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")
    
    return imagenes, etiquetas

def cargar_dataset_completo(dataset_path, max_por_clase=None):
    """
    Carga el dataset completo de entrenamiento y validación
    """
    print("Cargando dataset de entrenamiento...")
    
    # Cargar imágenes de entrenamiento
    train_dogs_path = os.path.join(dataset_path, 'train', 'dogs')
    train_cats_path = os.path.join(dataset_path, 'train', 'cats')
    
    train_dogs, train_dogs_labels = cargar_imagenes(train_dogs_path, 1, max_por_clase)  # Perros = 1
    train_cats, train_cats_labels = cargar_imagenes(train_cats_path, 0, max_por_clase)  # Gatos = 0
    
    X_train = train_dogs + train_cats
    y_train = train_dogs_labels + train_cats_labels
    
    print("\nCargando dataset de validación...")
    
    # Cargar imágenes de validación
    val_dogs_path = os.path.join(dataset_path, 'validation', 'dogs')
    val_cats_path = os.path.join(dataset_path, 'validation', 'cats')
    
    val_dogs, val_dogs_labels = cargar_imagenes(val_dogs_path, 1, max_por_clase)
    val_cats, val_cats_labels = cargar_imagenes(val_cats_path, 0, max_por_clase)
    
    X_val = val_dogs + val_cats
    y_val = val_dogs_labels + val_cats_labels
    
    # Convertir a arrays numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print(f"\nResumen del dataset:")
    print(f"Entrenamiento: {X_train.shape[0]} imágenes")
    print(f"Validación: {X_val.shape[0]} imágenes")
    print(f"Dimensionalidad por imagen: {X_train.shape[1]} características")
    print(f"Distribución de clases en entrenamiento: {np.unique(y_train, return_counts=True)}")
    print(f"Distribución de clases en validación: {np.unique(y_val, return_counts=True)}")
    
    return X_train, X_val, y_train, y_val

# Cargar el dataset (puedes ajustar max_por_clase para usar menos imágenes)
print("=" * 60)
print("CARGANDO DATASET PARA REGRESIÓN LOGÍSTICA")
print("=" * 60)

X_train, X_val, y_train, y_val = cargar_dataset_completo(dataset_path, max_por_clase=1000)

# Verificar que tenemos datos
if len(X_train) == 0 or len(X_val) == 0:
    print("Error: No se pudieron cargar imágenes. Verifica las rutas.")
    exit()

# Preprocesamiento: Escalar características
print("\nEscalando características...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Crear y entrenar el modelo de regresión logística
print("\n" + "=" * 60)
print("ENTRENANDO MODELO DE REGRESIÓN LOGÍSTICA")
print("=" * 60)

model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='liblinear',
    verbose=1
)

print("Iniciando entrenamiento...")
model.fit(X_train_scaled, y_train)

print("¡Entrenamiento completado!")

# Evaluar el modelo
print("\n" + "=" * 60)
print("EVALUANDO MODELO")
print("=" * 60)

# Predicciones
y_pred = model.predict(X_val_scaled)
y_pred_proba = model.predict_proba(X_val_scaled)

# Métricas
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print(f"\nMétricas finales en validación:")
print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"Precisión (Precision): {precision:.4f}")
print(f"Sensibilidad (Recall): {recall:.4f}")

# Reporte de clasificación
print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 60)
print(classification_report(y_val, y_pred, target_names=['Gatos', 'Perros']))

# Matriz de confusión
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Gatos', 'Perros'],
            yticklabels=['Gatos', 'Perros'])
plt.title('Matriz de Confusión - Regresión Logística')
plt.ylabel('Real')
plt.xlabel('Predicho')

# Gráfico de coeficientes (características más importantes)
plt.subplot(1, 2, 2)
coef_importancia = np.abs(model.coef_[0])
top_indices = np.argsort(coef_importancia)[-20:]  # Top 20 características

plt.barh(range(len(top_indices)), coef_importancia[top_indices])
plt.title('Top 20 Características Más Importantes')
plt.xlabel('Importancia (valor absoluto de coeficientes)')
plt.tight_layout()
plt.show()

# Función para predecir una imagen nueva
def predecir_imagen(ruta_imagen, modelo, escalador):
    """
    Función para predecir si una imagen es perro o gato usando regresión logística
    """
    try:
        # Cargar y preprocesar imagen
        img = Image.open(ruta_imagen).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)
        
        # Escalar
        img_scaled = escalador.transform(img_flat)
        
        # Predecir
        prediccion_proba = modelo.predict_proba(img_scaled)[0]
        prediccion_clase = modelo.predict(img_scaled)[0]
        
        clase = "Perro" if prediccion_clase == 1 else "Gato"
        probabilidad = prediccion_proba[1] if prediccion_clase == 1 else prediccion_proba[0]
        
        # Mostrar resultado
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f'Predicción: {clase}\nProbabilidad: {probabilidad:.2%}')
        plt.axis('off')
        plt.show()
        
        print(f"Probabilidades: [Gato: {prediccion_proba[0]:.2%}, Perro: {prediccion_proba[1]:.2%}]")
        
        return clase, probabilidad
        
    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        return None, None

# Información sobre la regresión logística
print("\n" + "=" * 60)
print("INFORMACIÓN SOBRE REGRESIÓN LOGÍSTICA")
print("=" * 60)
print("La regresión logística modela la probabilidad usando la función sigmoide:")
print("P(y=1) = 1 / (1 + e^(-(w·x + b)))")
print("\nCaracterísticas del modelo:")
print(f"- Número de características: {model.coef_.shape[1]}")
print(f"- Coeficientes (pesos): {model.coef_.shape}")
print(f"- Término de intercepción (sesgo): {model.intercept_[0]:.4f}")

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_regresion_logistica.pkl')
joblib.dump(scaler, 'escalador.pkl')
print("\nModelo guardado como 'modelo_regresion_logistica.pkl'")
print("Escalador guardado como 'escalador.pkl'")

# Ejemplo de uso (descomenta si tienes una imagen para probar)
# clase, prob = predecir_imagen("ruta/a/tu/imagen.jpg", model, scaler)
# if clase:
#     print(f"Resultado: {clase} con {prob:.2%} de probabilidad")
