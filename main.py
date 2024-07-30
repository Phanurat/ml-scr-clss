import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# อ่านข้อมูล
df = pd.read_csv('data/data.csv')

# เตรียมข้อมูล
X = df['text']
y = df['label']

# แปลงข้อความเป็นฟีเจอร์
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# สร้างและฝึกโมเดล
model = MultinomialNB()
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# แสดงผลลัพธ์
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
