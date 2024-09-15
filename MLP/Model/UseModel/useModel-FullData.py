import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

# โหลดโมเดลที่บันทึกไว้
model = tf.keras.models.load_model('trained_model.h5')

# โหลด scaler ที่บันทึกไว้
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ข้อมูลใหม่ที่ต้องการทำนาย
date_str = '2024-09-01'  # วันที่ที่ต้องการทำนาย
open_price = 220.00       # ราคาเปิด
high_price = 221.13       # ราคาสูงสุด
low_price = 219.18         # ราคาต่ำสุด
volume = 43617462         # ปริมาณการซื้อขาย
dividends = 0.00          # เงินปันผล
stock_splits = 0.00       # การแตกหุ้น

# แปลงวันที่เป็นตัวเลข
date = pd.to_datetime(date_str).toordinal()

# สร้างอาร์เรย์ข้อมูลใหม่
new_data = np.array([[date, open_price, high_price, low_price, volume, dividends, stock_splits]])

# ทำการมาตรฐานข้อมูลใหม่ตามที่ใช้ในการฝึก
new_data = scaler.transform(new_data)

# ทำนายราคาปิด
prediction = model.predict(new_data)
print("Prediction for new data:", prediction[0][0])
