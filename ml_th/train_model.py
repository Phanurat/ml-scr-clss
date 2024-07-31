import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize

# อ่านไฟล์ .csv
data = pd.read_csv('data/news.csv')

# กำหนด stop words ภาษาไทย
thai_stopwords = set(thai_stopwords())

# ฟังก์ชันในการทำความสะอาดข้อความและลบ stop words
def preprocess_text(text):
    tokens = word_tokenize(text, engine='newmm')  # ใช้ engine 'newmm' ในการตัดคำ
    tokens = [word for word in tokens if word not in thai_stopwords]  # ลบ stop words
    return ' '.join(tokens)

# ทำ preprocessing กับข้อมูล
data['text'] = data['text'].apply(preprocess_text)

# แยกข้อมูลเป็น training และ testing
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# ใช้ TfidfVectorizer ในการแปลงข้อความเป็นตัวเลข
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# สร้างและฝึกโมเดล Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ทำนายและประเมินผล
y_pred = model.predict(X_test_vec)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ใช้ Cross-validation เพื่อประเมินโมเดล
scores = cross_val_score(model, vectorizer.transform(data['text']), data['label'], cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

# ลองใช้โมเดล RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_vec, y_train)
y_pred_rf = model_rf.predict(X_test_vec)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ลองใช้โมเดล GradientBoostingClassifier
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train_vec, y_train)
y_pred_gb = model_gb.predict(X_test_vec)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print(classification_report(y_test, y_pred_gb))
