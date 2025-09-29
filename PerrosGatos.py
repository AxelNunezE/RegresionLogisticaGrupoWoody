import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración de parámetros
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 15

# Ruta de la carpeta con imágenes (debe tener subcarpetas 'train' y 'validation')
# Estructura esperada:
# dataset/
#   ├── train/
#   │   ├── dogs/
#   │   └── cats/
#   └── validation/
#       ├── dogs/
#       └── cats/

dataset_path = "perrosygatos.zip"  # Cambia esta ruta

# Crear generadores de datos con aumento de datos
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Cargar imágenes
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Binary classification (perros vs gatos)
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Clases: {train_generator.class_indices}")
print(f"Número de imágenes de entrenamiento: {train_generator.samples}")
print(f"Número de imágenes de validación: {validation_generator.samples}")

# Crear modelo con función de activación sigmoidal
model = keras.Sequential([
    # Capa convolucional 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),

    # Capa convolucional 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Capa convolucional 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Capa convolucional 4
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Aplanar para fully connected
    layers.Flatten(),
    layers.Dropout(0.5),

    # Capa densa con activación relu
    layers.Dense(512, activation='relu'),

    # Capa de salida con activación SIGMOIDAL (para clasificación binaria)
    layers.Dense(1, activation='sigmoid')  # Sigmoidal para probabilidad
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Para clasificación binaria
    metrics=['accuracy', 'precision', 'recall']
)

# Mostrar arquitectura del modelo
model.summary()

# Callbacks para mejorar el entrenamiento
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
]

# Entrenar el modelo
print("\nIniciando entrenamiento...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Evaluar el modelo
print("\nEvaluando el modelo...")
validation_loss, validation_accuracy, validation_precision, validation_recall = model.evaluate(validation_generator)
print(f"\nMétricas finales en validación:")
print(f"Pérdida (Loss): {validation_loss:.4f}")
print(f"Exactitud (Accuracy): {validation_accuracy:.4f}")
print(f"Precisión (Precision): {validation_precision:.4f}")
print(f"Sensibilidad (Recall): {validation_recall:.4f}")

# Hacer predicciones
print("\nGenerando predicciones...")
validation_generator.reset()
predictions = model.predict(validation_generator)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# Obtener las clases reales
true_classes = validation_generator.classes

# Métricas de clasificación
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN")
print("="*60)
print(classification_report(true_classes, predicted_classes,
                          target_names=validation_generator.class_indices.keys()))

# Matriz de confusión
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=validation_generator.class_indices.keys(),
            yticklabels=validation_generator.class_indices.keys())
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')

# Gráfico de precisión
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend()

plt.tight_layout()
plt.show()

# Función para predecir una imagen nueva
def predecir_imagen(ruta_imagen, modelo):
    """
    Función para predecir si una imagen es perro o gato
    """
    # Cargar y preprocesar imagen
    img = keras.preprocessing.image.load_img(ruta_imagen, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
    img_array /= 255.0  # Normalizar

    # Predecir
    prediccion = modelo.predict(img_array)[0][0]
    clase = "Perro" if prediccion > 0.5 else "Gato"
    probabilidad = prediccion if prediccion > 0.5 else 1 - prediccion

    # Mostrar resultado
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Predicción: {clase}\nProbabilidad: {probabilidad:.2%}')
    plt.axis('off')
    plt.show()

    return clase, probabilidad

# Ejemplo de uso (descomenta si tienes una imagen para probar)
# clase, prob = predecir_imagen("ruta/a/tu/imagen.jpg", model)
# print(f"Resultado: {clase} con {prob:.2%} de probabilidad")

# Guardar el modelo entrenado
model.save('modelo_perros_gatos.h5')
print("\nModelo guardado como 'modelo_perros_gatos.h5'")

# Mostrar información sobre la función sigmoidal
print("\n" + "="*60)
print("INFORMACIÓN SOBRE LA FUNCIÓN SIGMOIDAL")
print("="*60)
print("La función sigmoidal transforma cualquier número real en un valor entre 0 y 1")
print("Esto la hace ideal para clasificación binaria:")
print("- Valores cercanos a 0 → Gato")
print("- Valores cercanos a 1 → Perro")
print("- Umbral de decisión: 0.5")

# Ejemplo de salidas sigmoidales
print("\nEjemplos de salidas sigmoidales:")
ejemplos = [-5, -2, -0.5, 0, 0.5, 2, 5]
for x in ejemplos:
    sig = 1 / (1 + np.exp(-x))
    clase = "Perro" if sig > 0.5 else "Gato"
    print(f"Entrada: {x:5.1f} → Sigmoide: {sig:.4f} → Clase: {clase}")
