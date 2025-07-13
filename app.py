import streamlit as st
import joblib
import numpy as np

# 모델 및 인코더 불러오기
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("AI 기반 학습 성향 분석기")
st.write("당신의 공부 습관을 바탕으로 AI가 유형을 분석하고 시간표를 추천합니다!")

# 🟩 입력값 받기 함수
def user_input():
    col1 = st.selectbox("집중 잘 되는 시간대", encoders['하루 중 가장 집중이 잘 되는 시간대를 선택해 주세요'].classes_)
    col2 = st.selectbox("가장 졸린 시간대", encoders['하루 중 가장 졸린 시간대를 선택해 주세요'].classes_)
    col3 = st.selectbox("자신 있는 과목", encoders['공부할 때 좋아하거나 자신 있는 과목을 선택해 주세요'].classes_)
    col4 = st.selectbox("어려운 과목", encoders['가장 어려워하거나 회피하는 과목을 한가지 선택해 주세요'].classes_)
    col5 = st.selectbox("하루 순공 시간", encoders['평일 하루 순공시간을 선택해 주세요'].classes_)
    col6 = st.selectbox("집중 유지 시간", encoders['집중을 유지할 수 있는 시간을 선택해 주세요'].classes_)
    col7 = st.number_input("평일 평균 수면 시간 (숫자만 입력!)", min_value=0.0, max_value=12.0, step=0.5)
    col8 = st.selectbox("공부 계획 여부", encoders['평소에 공부 계획을 세우는 편인가요?'].classes_)

    encoded = [
        encoders['하루 중 가장 집중이 잘 되는 시간대를 선택해 주세요'].transform([col1])[0],
        encoders['하루 중 가장 졸린 시간대를 선택해 주세요'].transform([col2])[0],
        encoders['공부할 때 좋아하거나 자신 있는 과목을 선택해 주세요'].transform([col3])[0],
        encoders['가장 어려워하거나 회피하는 과목을 한가지 선택해 주세요'].transform([col4])[0],
        encoders['평일 하루 순공시간을 선택해 주세요'].transform([col5])[0],
        encoders['집중을 유지할 수 있는 시간을 선택해 주세요'].transform([col6])[0],
        col7,
        encoders['평소에 공부 계획을 세우는 편인가요?'].transform([col8])[0],
    ]

    return np.array(encoded).reshape(1, -1)

# ✅ 입력 먼저 받아두기
user_data = user_input()

# ✅ 버튼 클릭 시 예측 실행
if st.button("AI에게 내 학습 성향 분석받기"):
    prediction = model.predict(user_data)[0]

    st.success(f"예측 결과: 당신은 **군집 {prediction}번 유형**입니다!")

    if prediction == 0:
        st.info("**미라클 모닝형**: 오전 집중! 짧고 자주 쉬는 패턴 추천")
        st.image("cluster0.png", caption="추천 시간표 (미라클 모닝형)", use_container_width=True)

    elif prediction == 1:
        st.info("**밤 집중형**: 저녁~밤 집중! 자유로운 루틴")
        st.image("cluster1.png", caption="추천 시간표 (밤 집중형)", use_container_width=True)

    elif prediction == 2:
        st.info("**균형 집중형**: 오후~저녁에 고르게 분산 학습")
        st.image("cluster2.png", caption="추천 시간표 (균형 집중형)", use_container_width=True)

    elif prediction == 3:
        st.info("**계획적 저녁형**: 철저한 시간 계획 + 저녁 집중 전략")
        st.image("cluster3.png", caption="추천 시간표 (계획적 저녁형)", use_container_width=True)

    else:
        st.warning("분류되지 않은 유형입니다. 데이터가 부족할 수 있어요.")
