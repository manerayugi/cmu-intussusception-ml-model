"""Streamlit app: predicts the probability that non-operative reduction of
intussusception will succeed, using a pre-trained XGBoost model.

Run with: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "xgb_kmeans_smote.joblib"

# Feature vector expected by the model, in training order.
FEATURE_TEMPLATE = [
    "Type of intussusception_appendicocolic", "Type of intussusception_colocolic",
    "Type of intussusception_ileocolic", "Type of intussusception_ileoileal",
    "Type of intussusception_ileoileocolic",
    "Gender_Female", "Gender_Male",
    "Vomiting_absent", "Vomiting_present",
    "Bloody stool/Rectal bleeding_absent", "Bloody stool/Rectal bleeding_present",
    "Abdominal distension_absent", "Abdominal distension_present",
    "Palpable abdominal mass_absent", "Palpable abdominal mass_present",
    "Location of intussusception_LLQ", "Location of intussusception_LUQ",
    "Location of intussusception_RLQ", "Location of intussusception_RUQ",
    "Location of intussusception_rectum",
    "Method of reduction_Hydrostatic reduction", "Method of reduction_Pneumatic reduction",
    "Small bowel obstruction from Plain film Abdomen _No",
    "Small bowel obstruction from Plain film Abdomen _Yes",
    "a thick peripheral hypoechoic rim_absent", "a thick peripheral hypoechoic rim_present",
    "free intraperitoneum fluid_absent", "free intraperitoneum fluid_present",
    "fluid trapped within intussusceptum_absent", "fluid trapped within intussusceptum_present",
    "enlarged lymph node in intussusception_absent", "enlarged lymph node in intussusception_present",
    "pathologic lead point_absent", "pathologic lead point_present",
    "absence of blood flow in the intussusception_absent",
    "absence of blood flow in the intussusception_present",
    "Age", "Body weight", "Duration of symptoms", "Body Temperature",
]

# field -> allowed values (used both to render the selectbox and to validate
# that the resulting one-hot column exists in FEATURE_TEMPLATE)
CATEGORICAL_FIELDS = {
    "Type of intussusception": ["ileocolic", "colocolic", "ileoileocolic", "ileoileal", "appendicocolic"],
    "Gender": ["Male", "Female"],
    "Vomiting": ["present", "absent"],
    "Bloody stool/Rectal bleeding": ["present", "absent"],
    "Abdominal distension": ["present", "absent"],
    "Palpable abdominal mass": ["present", "absent"],
    "Location of intussusception": ["RUQ", "LUQ", "RLQ", "LLQ", "rectum"],
    "Method of reduction": ["Pneumatic reduction", "Hydrostatic reduction"],
    "Small bowel obstruction from Plain film Abdomen ": ["Yes", "No"],
    "a thick peripheral hypoechoic rim": ["present", "absent"],
    "free intraperitoneum fluid": ["present", "absent"],
    "fluid trapped within intussusceptum": ["present", "absent"],
    "enlarged lymph node in intussusception": ["present", "absent"],
    "pathologic lead point": ["present", "absent"],
    "absence of blood flow in the intussusception": ["present", "absent"],
}

NUMERIC_FIELDS = {
    # field: (label, min, max, default, step)
    "Age": ("อายุ (เดือน) / Age (months)", 0.0, 216.0, 11.0, 1.0),
    "Body weight": ("น้ำหนักตัว (กก.) / Body weight (kg)", 0.0, 100.0, 10.0, 0.1),
    "Duration of symptoms": ("ระยะเวลาที่มีอาการ (ชม.) / Duration of symptoms (hours)", 0.0, 500.0, 24.0, 1.0),
    "Body Temperature": ("อุณหภูมิกาย (°C) / Body temperature (°C)", 30.0, 42.0, 36.7, 0.1),
}

VALIDATION_NOTE = (
    "ผลจากการทดสอบภายในชุดข้อมูล 51 เคส (สถาบันเดียว): "
    "Accuracy 94.1%, Sensitivity (ทำนายสำเร็จถูกต้อง) 91.7%, "
    "Specificity (ทำนายล้มเหลวถูกต้อง) 100%, Precision 100%, F1-score 95.7% "
    "โมเดลนี้ไม่เคยทำนายว่า 'สำเร็จ' ผิดพลาดในเคสที่ล้มเหลวจริงในชุดทดสอบนี้ "
    "แต่เนื่องจากชุดข้อมูลมีขนาดเล็กและมาจากสถาบันเดียว ตัวเลขนี้จึงเป็นเพียงข้อมูลอ้างอิง "
    "ไม่ใช่การรับประกันความแม่นยำเมื่อใช้กับผู้ป่วยรายใหม่"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    row = {feature: 0 for feature in FEATURE_TEMPLATE}
    for field in CATEGORICAL_FIELDS:
        column = f"{field}_{inputs[field]}"
        if column not in row:
            raise ValueError(f"ค่าที่ไม่รู้จักสำหรับ '{field}': {inputs[field]!r}")
        row[column] = 1
    for field in NUMERIC_FIELDS:
        row[field] = inputs[field]
    return pd.DataFrame([row], columns=FEATURE_TEMPLATE)


def render_form() -> dict | None:
    inputs = {}
    with st.form("patient_form"):
        st.subheader("ข้อมูลพื้นฐาน (Basic information)")
        col1, col2 = st.columns(2)
        with col1:
            inputs["Gender"] = st.selectbox("Gender", CATEGORICAL_FIELDS["Gender"])
            label, lo, hi, default, step = NUMERIC_FIELDS["Age"]
            inputs["Age"] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)
            label, lo, hi, default, step = NUMERIC_FIELDS["Body weight"]
            inputs["Body weight"] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)
        with col2:
            label, lo, hi, default, step = NUMERIC_FIELDS["Body Temperature"]
            inputs["Body Temperature"] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)
            label, lo, hi, default, step = NUMERIC_FIELDS["Duration of symptoms"]
            inputs["Duration of symptoms"] = st.number_input(label, min_value=lo, max_value=hi, value=default, step=step)

        st.subheader("อาการและการตรวจร่างกาย (Symptoms & physical exam)")
        col1, col2 = st.columns(2)
        with col1:
            inputs["Vomiting"] = st.selectbox("Vomiting", CATEGORICAL_FIELDS["Vomiting"])
            inputs["Bloody stool/Rectal bleeding"] = st.selectbox(
                "Bloody stool / Rectal bleeding", CATEGORICAL_FIELDS["Bloody stool/Rectal bleeding"])
        with col2:
            inputs["Abdominal distension"] = st.selectbox(
                "Abdominal distension", CATEGORICAL_FIELDS["Abdominal distension"])
            inputs["Palpable abdominal mass"] = st.selectbox(
                "Palpable abdominal mass", CATEGORICAL_FIELDS["Palpable abdominal mass"])

        st.subheader("ผลภาพถ่ายรังสีและอัลตราซาวด์ (Imaging findings)")
        col1, col2 = st.columns(2)
        with col1:
            inputs["Type of intussusception"] = st.selectbox(
                "Type of intussusception", CATEGORICAL_FIELDS["Type of intussusception"])
            inputs["Location of intussusception"] = st.selectbox(
                "Location of intussusception", CATEGORICAL_FIELDS["Location of intussusception"])
            inputs["Small bowel obstruction from Plain film Abdomen "] = st.selectbox(
                "Small bowel obstruction (plain film)",
                CATEGORICAL_FIELDS["Small bowel obstruction from Plain film Abdomen "])
            inputs["a thick peripheral hypoechoic rim"] = st.selectbox(
                "Thick peripheral hypoechoic rim", CATEGORICAL_FIELDS["a thick peripheral hypoechoic rim"])
        with col2:
            inputs["free intraperitoneum fluid"] = st.selectbox(
                "Free intraperitoneal fluid", CATEGORICAL_FIELDS["free intraperitoneum fluid"])
            inputs["fluid trapped within intussusceptum"] = st.selectbox(
                "Fluid trapped within intussusceptum", CATEGORICAL_FIELDS["fluid trapped within intussusceptum"])
            inputs["enlarged lymph node in intussusception"] = st.selectbox(
                "Enlarged lymph node", CATEGORICAL_FIELDS["enlarged lymph node in intussusception"])
            inputs["pathologic lead point"] = st.selectbox(
                "Pathologic lead point", CATEGORICAL_FIELDS["pathologic lead point"])
            inputs["absence of blood flow in the intussusception"] = st.selectbox(
                "Absence of blood flow (Doppler)",
                CATEGORICAL_FIELDS["absence of blood flow in the intussusception"])

        st.subheader("วิธีการรักษาที่วางแผน (Planned reduction method)")
        inputs["Method of reduction"] = st.selectbox("Method of reduction", CATEGORICAL_FIELDS["Method of reduction"])

        submitted = st.form_submit_button("ทำนายผล / Predict", use_container_width=True)

    return inputs if submitted else None


def render_result(inputs: dict) -> None:
    try:
        model = load_model()
        x = build_feature_vector(inputs)
        failed_proba, success_proba = model.predict_proba(x)[0]
    except FileNotFoundError:
        st.error("ไม่พบไฟล์โมเดล (models/xgb_kmeans_smote.joblib) กรุณาตรวจสอบการติดตั้ง")
        return
    except Exception as exc:
        st.error(f"เกิดข้อผิดพลาดระหว่างการคำนวณผล: {exc}")
        return

    st.divider()
    st.subheader("ผลการทำนาย (Prediction result)")
    col1, col2 = st.columns(2)
    col1.metric("โอกาสสำเร็จ (Success)", f"{success_proba * 100:.1f}%")
    col2.metric("โอกาสล้มเหลว (Failed)", f"{failed_proba * 100:.1f}%")
    st.progress(float(success_proba))
    st.caption(
        "ผลลัพธ์นี้เป็นเครื่องมือช่วยสนับสนุนการตัดสินใจทางคลินิกเท่านั้น "
        "ไม่ใช่การวินิจฉัยหรือคำแนะนำทางการแพทย์ และไม่สามารถทดแทนดุลยพินิจของแพทย์ผู้ดูแลได้ "
        "This tool is a clinical decision-support aid only, not a diagnosis or medical "
        "recommendation, and does not replace clinical judgment."
    )


def main():
    st.set_page_config(page_title="Intussusception Reduction Predictor", page_icon="🏥", layout="centered")
    st.title("🏥 Machine Learning Aid for Predicting Non-operative Reduction of Intussusception")
    st.caption(
        "กรอกข้อมูลผู้ป่วยด้านล่าง แล้วกด 'ทำนายผล' เพื่อประเมินโอกาสสำเร็จ/ล้มเหลว "
        "ของการลดกลืนของลำไส้แบบไม่ผ่าตัด (non-operative reduction)"
    )

    inputs = render_form()
    if inputs is not None:
        render_result(inputs)

    with st.expander("เกี่ยวกับโมเดลนี้ (About this model)"):
        st.write(f"**โมเดล:** XGBoost (oversampled ด้วย K-means SMOTE)")
        st.write(VALIDATION_NOTE)


if __name__ == "__main__":
    main()
