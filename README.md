# ML Intussusception Reduction Predictor

เว็บแอปสำหรับแพทย์ ใช้ทำนายโอกาสสำเร็จ/ล้มเหลวของการลดกลืนของลำไส้แบบไม่ผ่าตัด
(non-operative reduction of intussusception) จากข้อมูลผู้ป่วย โดยใช้โมเดล Machine
Learning ที่ฝึกไว้แล้ว

แอปนี้เป็น Streamlit แอปเดี่ยว โหลดโมเดลจากไฟล์ `.joblib` ตรงๆ ไม่ต้องพึ่งบริการ API แยก

## วิธีติดตั้งและรัน (Setup & run)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ URL ที่ Streamlit แสดง (ค่าเริ่มต้น http://localhost:8501)

## โมเดลที่ใช้ (Model)

- **XGBoost classifier**, oversampled ด้วย K-means SMOTE ระหว่างเทรน
- ไฟล์โมเดล: `models/xgb_kmeans_smote.joblib`
- ผลการทดสอบภายใน (internal test set, n=51, สถาบันเดียว):
  Accuracy 94.1%, Sensitivity 91.7%, Specificity 100%, Precision 100%, F1-score 95.7%

**ข้อจำกัดที่ควรทราบ**: ตัวเลขข้างต้นมาจากชุดข้อมูลทดสอบขนาดเล็กของสถาบันเดียว และในขั้นตอนฝึกโมเดล
มีการทำ oversampling (K-means SMOTE) ก่อนแบ่ง train/test ซึ่งอาจทำให้ผลลัพธ์ดูดีกว่าความเป็นจริงเล็กน้อย
(data leakage between train/test) ตัวเลขนี้จึงควรใช้เป็นข้อมูลอ้างอิงคร่าวๆ ไม่ใช่ตัวชี้วัดที่แม่นยำสมบูรณ์
โมเดลนี้เป็นเครื่องมือช่วยสนับสนุนการตัดสินใจ (decision support) ไม่ใช่เครื่องมือวินิจฉัยหรือทดแทนดุลยพินิจแพทย์

โมเดลถูกฝึกและทดลองในโปรเจกต์แยกต่างหาก (`ml-intussusception-model`) — repo นี้ใช้เฉพาะไฟล์โมเดล
ที่ฝึกเสร็จแล้วมาให้บริการเท่านั้น ไม่มีโค้ดฝึกโมเดลอยู่ในนี้

## โครงสร้างไฟล์

- `app.py` — Streamlit app หลัก (UI + การเรียกใช้โมเดล)
- `models/xgb_kmeans_smote.joblib` — โมเดลที่ใช้งานจริง
- `requirements.txt` — ไลบรารีที่ต้องติดตั้ง (เวอร์ชันถูก pin ให้ตรงกับไฟล์โมเดลเพื่อความเสถียร)

## Deploy

ใช้ deploy บน Streamlit Community Cloud, หรือเซิร์ฟเวอร์ภายในโรงพยาบาลที่รัน Python ได้ก็ได้
ไม่ต้องเปิดพอร์ตหรือรันบริการเพิ่มเติมใดๆ นอกจาก `streamlit run app.py`
