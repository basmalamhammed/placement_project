from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="توقع التعيين بعد الجامعة", page_icon="🎓")

# كل الصفحة من اليمين لليسار
st.markdown(
    """
    <style>
    .stApp, .stMarkdown, label, p, h1, h2, h3 { direction: rtl; text-align: right; }
    div[data-baseweb="select"] > div { direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 هتتعيّن بعد التخرج؟ ولا زيي")
st.write("اكتب بياناتك الدراسية، والموديل هيقولك فرصتك في التعيين كام في المية.")
st.caption("الموديل متدرّب على بيانات طلاب جامعيين من حرم جامعي في الهند، فالنتيجة تقدير تقريبي مش وعد.")

if not (MODELS_DIR / "placement_model.joblib").exists():
    st.error("مفيش موديل متسيف. شغّل الأول:  python train.py")
    st.stop()

model = joblib.load(MODELS_DIR / "placement_model.joblib")
best_name = (MODELS_DIR / "best_model_name.txt").read_text()

# عربي ← القيمة اللي الموديل فاهمها
GENDER = {"ذكر": "M", "أنثى": "F"}
STREAM = {"علمي": "Science", "تجاري": "Commerce", "أدبي": "Arts"}
DEGREE_TYPE = {"علوم وتكنولوجيا (زي علوم الحاسب)": "Sci&Tech", "إدارة وتجارة": "Comm&Mgmt", "أخرى": "Others"}
YES_NO = {"لأ": "No", "أيوه": "Yes"}
BOARD = {"مركزي": "Central", "آخر": "Others"}

tab1, tab2 = st.tabs([" توقّع", " مقارنة الموديلات"])

# ---------------- التاب الأول: التوقع ----------------
with tab1:
    st.subheader("بياناتك")

    gender = st.radio("النوع", list(GENDER), horizontal=True)

    c1, c2 = st.columns(2)
    with c1:
        ssc_p = st.number_input("نسبتك في الإعدادية (%)", 0.0, 100.0, 80.0, step=1.0)
        hsc_p = st.number_input("نسبتك في الثانوية (%)", 0.0, 100.0, 80.0, step=1.0)
    with c2:
        degree_p = st.number_input(
            "نسبتك في الجامعة (%)", 0.0, 100.0, 70.0, step=1.0,
            help="المعدل التراكمي لحد دلوقتي بالنسبة المئوية",
        )
        hsc_s = st.selectbox("قسم الثانوية", list(STREAM))

    degree_t = st.selectbox("نوع الدراسة الجامعية", list(DEGREE_TYPE))
    workex = st.radio(
        "عندك خبرة شغل أو تدريب عملي؟", list(YES_NO), horizontal=True,
        help="تدريب صيفي أو شغل جزئي أو فريلانس",
    )

    has_etest = st.checkbox("عملت اختبار قدرات / توظيف؟")
    etest_p = np.nan  # لو ماعملتوش، الموديل بيحط قيمة متوسطة مكانه
    if has_etest:
        etest_p = st.number_input(
            "نتيجة اختبار القدرات (%)", 0.0, 100.0, 70.0, step=1.0,
            help="اختبار أولي بتعمله الكلية أو الشركات: منطق وحساب ولغة",
        )

    with st.expander("بيانات إضافية (اختياري — سيبها زي ما هي لو مش متأكد)"):
        ssc_b = st.selectbox("نوع منهج الإعدادية", list(BOARD))
        hsc_b = st.selectbox("نوع منهج الثانوية", list(BOARD))

    if st.button("احسب فرصتي", type="primary"):
        row = pd.DataFrame([{
            "gender": GENDER[gender], "ssc_p": ssc_p, "ssc_b": BOARD[ssc_b],
            "hsc_p": hsc_p, "hsc_b": BOARD[hsc_b], "hsc_s": STREAM[hsc_s],
            "degree_p": degree_p, "degree_t": DEGREE_TYPE[degree_t],
            "workex": YES_NO[workex], "etest_p": etest_p,
        }])

        prob = model.predict_proba(row)[0][1]
        st.metric("فرصتك في التعيين", f"{prob:.0%}")
        st.progress(float(prob))

        if prob >= 0.7:
            st.success(" فرصتك كويسة جدًا! يبختك يا امنيه هتعيشي احسن عيشه")
        elif prob >= 0.5:
            st.info("🙂 فرصتك كويسة، وممكن تتحسن أكتر. يعني خلي عندك دم")
        else:
            st.warning(" فرصتك أقل من المتوسط. الخبرة العملية والتدريب بيفرقوا في الغالب يا عبيد السلايدات")

    st.caption(f"الموديل المستخدم: {best_name}")

# ---------------- التاب التاني: المقارنة ----------------
with tab2:
    st.write("دي نتيجة تجربة كل الخوارزميات على نفس البيانات. الأعلى أحسن.")

    results = pd.read_csv(MODELS_DIR / "results.csv")
    arabic = results.rename(columns={
        "Model": "الموديل",
        "Accuracy": "الدقة (Accuracy)",
        "Precision": "Precision",
        "Recall": "Recall",
        "F1": "F1",
        "ROC-AUC": "ROC-AUC",
        "CV F1": "F1 بعد 5 تجارب",
    })
    st.dataframe(arabic.round(3), hide_index=True)

    st.bar_chart(results.set_index("Model")[["F1", "CV F1"]])

    st.info(
        f"أفضل موديل حسب F1: **{best_name}**\n\n"
        "• **Precision**: من اللي قلنا هيتعينوا، كام فعلًا اتعيّن.\n\n"
        "• **Recall**: من اللي اتعيّنوا فعلًا، كام لقيناهم.\n\n"
        "• **F1**: توازن بين الاتنين."
    )
