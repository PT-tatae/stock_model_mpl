import yfinance as yf

# กำหนดสัญลักษณ์ของบริษัท
ticker = 'AAPL'  # ตัวอย่าง: Apple Inc.

# โหลดข้อมูลของบริษัท
company = yf.Ticker(ticker)

# ดึงข้อมูลงบกำไรขาดทุน
income_statement = company.financials
print("Income Statement:")
print(income_statement)

# ดึงข้อมูลงบดุล
balance_sheet = company.balance_sheet
print("\nBalance Sheet:")
print(balance_sheet)

# ดึงข้อมูลมูลค่าตลาด
market_cap = company.info['marketCap']
print(f"\nMarket Cap: {market_cap}")
