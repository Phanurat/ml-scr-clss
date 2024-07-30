import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# โหลดโมเดลและตัวแปลงฟีเจอร์
model = joblib.load('text_classification_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ข้อความที่ต้องการทำนาย
new_texts = [
    "นายกฯ ประกาศนโยบายใหม่ในการพัฒนาประเทศ",
    "พรรคก้าวไกลเตรียมประกาศผู้สมัครใหม่",
    "รัฐบาลจัดงานเทศกาลสำคัญที่กรุงเทพฯ"
]

# แปลงข้อความใหม่เป็นฟีเจอร์
X_new = vectorizer.transform(new_texts)

# ทำนายผล
predictions = model.predict(X_new)
for text, prediction in zip(new_texts, predictions):
    print(f"ข้อความ: {text}")
    print(f"การทำนาย: {prediction}")
