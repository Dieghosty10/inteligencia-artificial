import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Generando los datos")
x = np.linspace(-5, 5, 1000)
y = x**2 + np.random.normal(0, 2, 1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

print(f"Datos de entrenamiento: {x_train.shape[0]} muestras")
print("Datos listos para la red")

print("\n2. Construyendo el modelo")
model = Sequential([
    Dense(32, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.summary()

print("\n3. Compilando y entrenando el modelo")
model.compile(optimizer='adam',
              loss="mse",
              metrics=['mae'])

history = model.fit(x_train, y_train,
                    epochs=100,
                    validation_split=0.1,
                    verbose=0)
print("Entrenamiento finalizado")

print("\n4. Evaluando el modelo")
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Error Cuadrático Medio (MSE) en prueba: {loss:.2f}")
print(f"Error Absoluto Medio (MAE) en prueba: {mae:.2f}")

x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
predictions = model.predict(x_range, verbose=0)

print("\n5. Visualización resultados")
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label="Datos originales ($x$ vs $x^2 + ruido$)", alpha=0.5, color='red')
plt.plot(x_range, predictions, color='blue', linewidth=3, label="Curva de predicción")
plt.title(f"Regresión No Lineal con Red Neuronal Densa (MAE: {mae:.2f})")
plt.xlabel("Valor de Entrada ($X$)")
plt.ylabel("Valor de Salida ($Y$)")
plt.legend()
plt.grid(True)
plt.show()