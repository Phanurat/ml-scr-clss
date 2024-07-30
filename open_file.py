import pandas as pd

# อ่านไฟล์ CSV
df = pd.read_csv('data/new_data.csv')

# ดูข้อมูลเบื้องต้น
print(df.head())
