import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Generar los Datos")
x=np.linspace(-5,5,1000)
y=x**2+np.random.normal(0,2,1000)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
print(f"Datos de Entrenamiento: {x_train.shape[0]} Muestras")
print("Datos Listos para la Red")
print("\n2. Construyendo el Modelo ")
model=Sequential([
    Dense(32,activation='relu',input_shape=(1,)),
    Dense(32,activation='relu'),
    Dense (1)
])

model.summary()
print("\n3. Compilando y Entrenando el modelo ")
model.compile( optimizer='adam',
              loss='mse',
              metrics=['mae'])

history=model.fit(x_train,y_train,
                  epochs=100,
                  validation_split=0.1,
                  verbose=0)

print("Entrenamiento Finalizado ")
print("\n4. Evaluando el Modelo")
loss,mae=model.evaluate(x_test,y_test,verbose=0)
print(f"Error Cuadratico Medio (loss/mae) en prueba: {loss:.2f}")
print(f"Error Absoluto Medio (mae) en Prueba: {mae:.2f}")
x_range=np.linspace(x.min(),x.max(),100).reshape(-1,1)
Predictions= model.predict(x_range)
print("\n5. Visualizacion Resultados")
plt.figure(figsize=(10,6))
plt.scatter(x,y,label="Datos Originales (x vs x^2+ruido)", alpha=0.5,color='purple')
plt.plot(x_range,Predictions,color='orange',linewidth=3,label="Curva de Prediccion")
plt.title(f"Regrecion No Lineal Con Red Neuronal Densa(mae:{mae:.2f})")
plt.xlabel("Valor de Salida (x)")
plt.ylabel("Valor de Salida (y)")
plt.legend()
plt.grid(True)
plt.savefig("imagen.png", dpi=300, bbox_inches='tight')
plt.show
