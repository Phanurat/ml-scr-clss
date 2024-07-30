import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from imblearn.over_sampling import RandomOverSampler  # ใช้ RandomOverSampler แทน SMOTE

# โหลดข้อมูล
df = pd.read_csv('data/new_data.csv')

# ตรวจสอบการกระจายของข้อมูล
print("Class distribution before resampling:")
print(df['who'].value_counts())

# เตรียมข้อมูล
X = df['text']
y = df['who']

# แปลงข้อความเป็นฟีเจอร์
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# จัดการกับข้อมูลไม่สมดุล
ros = RandomOverSampler(random_state=42)  # ใช้ RandomOverSampler
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# สร้างและฝึกโมเดล Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# ทำนายผล
y_pred = model.predict(X_test)

# แสดงผลลัพธ์
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# บันทึกโมเดลและการแปลงฟีเจอร์
joblib.dump(model, 'new_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
