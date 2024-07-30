# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# อ่านข้อมูลจากไฟล์ CSV
df = pd.read_csv('data/test_data.csv', encoding='utf-8-sig')

# ข้อมูลและป้ายกำกับ
X = df['text']
y = df['who']

# แปลงข้อความเป็นเวกเตอร์ TF-IDF
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# แบ่งข้อมูลเป็นชุดฝึกอบรมและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.3, random_state=42)

# สร้างโมเดล Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# ทำนายข้อมูลทดสอบ
y_pred = model.predict(X_test)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# แสดงผลลัพธ์
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# ทดสอบการทำนายข้อความใหม่
def predict_new_text(text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

# ทดสอบการทำนาย
#new_text = 'บรรยากาศนายเศรษฐา ทวีสิน นายกรัฐมนตรี เป็นประธานพิธีทำบุญตักบาตรถวายพระราชกุศลเนื่องในโอกาสพระราชพิธีมหามงคลเฉลิมพระชนมพรรษา 6 รอบ 26 กรกฎาคม 2567 โดยมีองคมนตรี ภริยา รวมถึงคณะรัฐมนตรีร่วมในพิธีด้วย โดยหลังพิธีทำบุญตักบาตร นายเศรษฐา ทวีสิน นายกรัฐมนตรี เป็นผู้กล่าวนำข้าราชการทั้งหมดในพิธีถวายสัตย์ปฏิญาณฯ พร้อมทั้งร่วมร้องเพลงสรรเสริญพระบารมี และเพลงสดุดีจอมราชา ต่อหน้าพระบรมฉายาลักษณ์พระบาทสมเด็จพระเจ้าอยู่หัวฯ ณ มณฑลพิธีท้องสนามหลวง (เมื่อวันที่ 28 กรกฎาคม 2567)'

new_text = input("วิเคราะห์ข่าวเกี่ยวกับใคร : ==> ")

print(new_text)
print(f'Prediction for new text: {predict_new_text(new_text)}')
