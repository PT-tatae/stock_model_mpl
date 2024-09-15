import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

import matplotlib.pyplot as plt
# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv('stock_history.csv')

# แปลงวันที่เป็นตัวเลข
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

# กำหนด features และ target
X = df[['Date', 'Open',  'Volume']]
y = df['Close']

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# มาตรฐานข้อมูล
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# สร้างโมเดลที่มี Hidden Layer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train_scaled.shape[1],), activation='tanh'),  # Hidden Layer แรก
    tf.keras.layers.Dense(32, activation='tanh'),                                            # Hidden Layer ที่สอง
    tf.keras.layers.Dense(16, activation='tanh'),                                            # Hidden Layer ที่สาม
    tf.keras.layers.Dense(1)                                                                 # Output Layer 
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# ฝึกสอนโมเดล
history = model.fit(X_train_scaled, y_train, epochs=250, verbose=0, validation_data=(X_test_scaled, y_test))

# ประเมินผลโมเดล
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f'Loss: {loss:.4f}, MAE: {mae:.4f}')


# ดึงข้อมูลจาก history
history_dict = history.history

# สร้างกราฟแสดงค่า loss และ mae
epochs = range(1, len(history_dict['loss']) + 1)

plt.figure(figsize=(12, 6))

# กราฟ loss (MSE)
plt.subplot(1, 2, 1)
plt.plot(epochs, history_dict['loss'], 'bo-', label='Training loss')
plt.plot(epochs, history_dict['val_loss'], 'r-', label='Validation loss')
plt.title('Training and Validation Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

# กราฟ MAE
plt.subplot(1, 2, 2)
plt.plot(epochs, history_dict['mae'], 'bo-', label='Training MAE')
plt.plot(epochs, history_dict['val_mae'], 'r-', label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# บันทึกโมเดล
model.save('trained_model_ShotData.h5')
print("Model saved to ''trained_model_ShotData.h5'")

# บันทึก scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler saved to 'scaler.pkl'")
