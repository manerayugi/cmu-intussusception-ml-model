# 🏥 Machine Learning Model Aid Prediction for Failed Nonoperative Reduction of Intussusception

เว็บแอปสำหรับแพทย์ ใช้ทำนายโอกาสสำเร็จ/ล้มเหลวของการลดกลืนของลำไส้แบบไม่ผ่าตัด
(non-operative reduction of intussusception) จากข้อมูลผู้ป่วย โดยใช้โมเดล Machine Learning
ที่ฝึกไว้แล้ว

- 🎯 **กรอกข้อมูลผู้ป่วย → ได้ % โอกาสสำเร็จ/ล้มเหลวทันที**
- 🧩 **Streamlit แอปเดี่ยว** โหลดโมเดลจากไฟล์ `.joblib` ตรงๆ ไม่ต้องพึ่ง API แยก ไม่มีจุดเสีย (network hop) เพิ่ม
- 📝 **มี log ทุกครั้งที่ทำนาย** (input, output, เวลาที่ใช้) ทั้งในเทอร์มินัลและไฟล์ `logs/predictions.log`
- ⚠️ **ไม่ใช่เครื่องมือวินิจฉัย** — เป็นแค่ตัวช่วยสนับสนุนการตัดสินใจ ไม่ทดแทนดุลยพินิจแพทย์

## 🚀 วิธีติดตั้งและรัน (Setup & run)

`requirements.txt` pin เวอร์ชันตรงกับ conda env `personal` ของเครื่องนี้ (Python 3.14)
ซึ่งตรงกับเวอร์ชันที่ Streamlit Community Cloud ใช้จริงด้วย ดังนั้นรันตรงกับ env ส่วนตัวได้เลย
ไม่ต้องสร้าง virtual env แยก:

```bash
conda activate personal
pip install -r requirements.txt
streamlit run app.py
```

ถ้าอยากแยก environment ต่างหาก (เช่น ใช้เครื่องอื่น หรือกลัวชนกับ package อื่นที่ติดตั้งไว้)
ก็สร้าง venv ก่อนได้ตามปกติ:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ URL ที่ Streamlit แสดง (ค่าเริ่มต้น http://localhost:8501)

## 🧠 โมเดลที่ใช้ (Model)

- **XGBoost classifier**, oversampled ด้วย K-means SMOTE ระหว่างเทรน
- ไฟล์โมเดล: `models/xgb_kmeans_smote.joblib`
- ผลการทดสอบภายใน (internal test set, n=51, สถาบันเดียว):
  - Accuracy **94.1%**
  - Sensitivity **91.7%**
  - Specificity **100%**
  - Precision **100%**
  - F1-score **95.7%**

> ⚠️ **ข้อจำกัดที่ควรทราบ**: ตัวเลขข้างต้นมาจากชุดข้อมูลทดสอบขนาดเล็กของสถาบันเดียว และในขั้นตอนฝึกโมเดล
> มีการทำ oversampling (K-means SMOTE) ก่อนแบ่ง train/test ซึ่งอาจทำให้ผลลัพธ์ดูดีกว่าความเป็นจริงเล็กน้อย
> (data leakage ระหว่าง train/test) ตัวเลขนี้จึงควรใช้เป็นข้อมูลอ้างอิงคร่าวๆ ไม่ใช่ตัวชี้วัดที่แม่นยำสมบูรณ์
> โมเดลนี้เป็นเครื่องมือช่วยสนับสนุนการตัดสินใจ (decision support) เท่านั้น

โมเดลถูกฝึกและทดลองในโปรเจกต์แยกต่างหาก (`ml-intussusception-model`) — repo นี้ใช้เฉพาะไฟล์โมเดล
ที่ฝึกเสร็จแล้วมาให้บริการเท่านั้น ไม่มีโค้ดฝึกโมเดลอยู่ในนี้

## 📂 โครงสร้างไฟล์

- `app.py` — Streamlit app หลัก (form กรอกข้อมูล + เรียกใช้โมเดล + แสดงผล)
- `models/xgb_kmeans_smote.joblib` — โมเดลที่ใช้งานจริง
- `requirements.txt` — ไลบรารี Python (เวอร์ชัน pin ตายตัว ให้ตรงกับที่ทดสอบกับไฟล์โมเดล)
- `packages.txt` — system package (`libgomp1`) ที่ XGBoost ต้องใช้บน Linux/Streamlit Cloud
- `runtime.txt` — pin เวอร์ชัน Python สำหรับ Streamlit Community Cloud
- `logs/` — log การทำนายแต่ละครั้ง (ไม่ commit เข้า git, สร้างอัตโนมัติตอนรัน)

## ☁️ Deploy บน Streamlit Community Cloud

1. Push repo นี้ขึ้น GitHub (ต้องมี `app.py`, `requirements.txt`, `packages.txt`, `runtime.txt`, `models/xgb_kmeans_smote.joblib`)
2. ไปที่ [share.streamlit.io](https://share.streamlit.io) → New app → เลือก repo/branch/`app.py`
3. Deploy ได้เลย ไม่ต้องตั้งค่า secrets หรือเปิดพอร์ตเพิ่ม

**🐛 ถ้า deploy แล้ว error**:
- `libgomp.so.1: cannot open shared object file` → เช็คว่า `packages.txt` (มี `libgomp1`) อยู่ที่ root ของ repo จริง แล้ว reboot app
- ติดตั้ง dependency ไม่ผ่าน / เวอร์ชันชนกัน → เช็ค log ว่า Streamlit Cloud ใช้ Python เวอร์ชันไหนจริง (บรรทัด `Using Python ... environment at ...` — Cloud มักเมิน `runtime.txt` แล้วใช้เวอร์ชันล่าสุดของตัวเองแทน) ถ้าไม่ตรงกับที่ `requirements.txt` รองรับ ให้ไปตั้งค่า Python version ในหน้า **Settings ของแอป → Python version** (หรือลบแอปแล้ว deploy ใหม่ พร้อมเลือกเองใน "Advanced settings") ให้ตรงกับที่ log แสดงจริง แล้ว pin `requirements.txt`/`runtime.txt` ให้ตรงกับเวอร์ชันนั้น

> ℹ️ ตอน `joblib.load()` โมเดลบน Python 3.14 อาจเห็น `DeprecationWarning: Setting the shape on a NumPy array...`
> ใน log — เป็น warning จากภายใน joblib เอง (ยังไม่มีเวอร์ชันใหม่กว่าที่แก้) ไม่กระทบผลการทำนาย ปลอดภัยที่จะเพิกเฉย

## 🔒 หมายเหตุเรื่องข้อมูล

`logs/predictions.log` เก็บข้อมูลทางคลินิกของผู้ป่วยที่กรอกเข้ามา (ไม่มีชื่อ/HN แต่ยังเป็นข้อมูลสุขภาพ)
ถ้า deploy แบบ public ควรจำกัดสิทธิ์การเข้าถึง log นี้ หรือปิดการเก็บ log ถ้าไม่จำเป็น

## 📚 งานวิจัยที่เกี่ยวข้อง (References)

โมเดลและแนวคิดในแอปนี้มาจากงานวิจัยต่อไปนี้ สำหรับผู้ที่สนใจอ่านรายละเอียดเพิ่มเติม:

- Rinkaewngam, P., Chouvatut, V., & Khorana, J. *Machine Learning Model Aid Prediction for
  Failed Nonoperative Reduction of Intussusception*.
  [datascience.cmu.ac.th/storage/articles/44.pdf](https://datascience.cmu.ac.th/storage/articles/44.pdf)
- พิทยาธร รินแก้วงาม. (2566). *การใช้ตัวแบบของเครื่องจักรเรียนรู้เพื่อทำนายความล้มเหลวในการรักษาภาวะลำไส้กลืนกัน
  แบบไม่ผ่าตัด*. [ปริญญาโท, มหาวิทยาลัยเชียงใหม่]. บัณฑิตวิทยาลัย มหาวิทยาลัยเชียงใหม่, เชียงใหม่.
  [cmudc.library.cmu.ac.th/frontend/Info/item/dc:174030](https://cmudc.library.cmu.ac.th/frontend/Info/item/dc:174030)
