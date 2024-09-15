import   as yf
import pandas as pd
from datetime import datetime, timedelta

# กำหนดสัญลักษณ์ของบริษัท
ticker = 'AAPL'  # ตัวอย่าง: Apple Inc.

# โหลดข้อมูลของบริษัท
company = yf.Ticker(ticker)

# กำหนดช่วงเวลา 1 ปีที่ผ่านมา
end_date = datetime.today()
start_date = end_date - timedelta(days=365)

# ดึงข้อมูลราคาหุ้นย้อนหลัง 1 ปี
history = company.history(start=start_date, end=end_date)
history.to_csv('stock_history.csv')
print("Stock history saved to 'stock_history.csv'")

# ดึงข้อมูลงบกำไรขาดทุน
income_statement = company.financials
income_statement.to_csv('income_statement.csv')  # บันทึกลงในไฟล์ CSV
print("Income Statement saved to 'income_statement.csv'")

# ดึงข้อมูลงบดุล
balance_sheet = company.balance_sheet
balance_sheet.to_csv('balance_sheet.csv')  # บันทึกลงในไฟล์ CSV
print("Balance Sheet saved to 'balance_sheet.csv'")

# ดึงข้อมูลมูลค่าตลาด
market_cap = company.info['marketCap']
# บันทึกข้อมูลมูลค่าตลาดลงในไฟล์ CSV
market_cap_df = pd.DataFrame([{'Market Cap': market_cap}])
market_cap_df.to_csv('market_cap.csv', index=False)
print("Market Cap saved to 'market_cap.csv'")
