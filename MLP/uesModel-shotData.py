import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

# โหลดโมเดลที่บันทึกไว้
model = tf.keras.models.load_model('trained_model_ShotData.h5')

# โหลด scaler ที่บันทึกไว้
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ข้อมูลใหม่ที่ต้องการทำนาย
date_str = '2024-09-05'  # วันที่ที่ต้องการทำนาย
open_price = 220.85       # ราคาเปิด
volume = 43780100         # ปริมาณการซื้อขาย


# แปลงวันที่เป็นตัวเลข
date = pd.to_datetime(date_str).toordinal()

# สร้างอาร์เรย์ข้อมูลใหม่
new_data = np.array([[date, open_price, volume,]])

# ทำการมาตรฐานข้อมูลใหม่ตามที่ใช้ในการฝึก
new_data = scaler.transform(new_data)

# ทำนายราคาปิด
prediction = model.predict(new_data)
print("Prediction for new data:", prediction[0][0])
