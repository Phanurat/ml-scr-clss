from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# โหลดโมเดลและ Tokenizer ของ BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(df['who'].unique()))

# แปลงข้อความเป็นเวกเตอร์
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# เตรียมข้อมูล
from datasets import Dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# แบ่งข้อมูล
split_dataset = tokenized_dataset.train_test_split(test_size=0.3)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# ตั้งค่าการฝึกอบรม
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# ฝึกอบรมโมเดล
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# ประเมินโมเดล
results = trainer.evaluate()
print(results)

# ทดสอบการทำนาย
def predict_new_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    return prediction.item()

# ทดสอบการทำนาย
new_text = 'นายกฯ ประกาศแผนพัฒนาเศรษฐกิจใหม่'
print(f'Prediction for new text: {predict_new_text(new_text)}')
