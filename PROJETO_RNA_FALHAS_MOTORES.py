import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Carregar a base de dados
df = pd.read_csv(r"C:\Users\User\Desktop\Nova pasta\base_falhas_motores_simulada.csv")
print(df.head())

# 2. Separar entradas (X) e saída (y)
X = df[["corrente_rms", "freq_vibracao", "ruido_std"]].values
y = to_categorical(df["classe"].values, num_classes=3)  # One-hot encoding

# 3. Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 5. Criar o modelo de Rede Neural
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 6. Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# 8. Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia final no conjunto de teste: {accuracy*100:.2f}%")

# 9. Plotar gráficos de acurácia e perda
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda (Loss) por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()