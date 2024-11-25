import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# 파일 업로드
st.title("FBProphet Time Series Forecasting App")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    try:
        # CSV 파일을 데이터프레임으로 읽기
        df = pd.read_csv(uploaded_file)

        # 데이터 확인
        st.write("업로드된 데이터:")
        st.write(df.head())

        # 환자 선택 기능
        option = st.selectbox("환자번호 또는 환자이름으로 선택하세요", ["환자번호", "환자이름"])
        if option == "환자번호":
            patients = df["환자번호"].unique()
        else:
            patients = df["환자이름"].unique()

        selected_patient = st.selectbox(f"{option}을 선택하세요", patients)

        # 선택한 환자의 데이터 필터링
        if option == "환자번호":
            patient_data = df[df["환자번호"] == selected_patient]
        else:
            patient_data = df[df["환자이름"] == selected_patient]

        # 시계열 예측을 위한 데이터 준비
        patient_data['측정날짜'] = pd.to_datetime(patient_data['측정날짜'])
        patient_data = patient_data.sort_values('측정날짜')

        metrics = ["수축기혈압", "이완기혈압", "맥박", "혈당", "체온", "호흡", "체중"]
        
        for metric in metrics:
            st.subheader(f"{metric} 예측")
            
            # FBProphet 모델을 사용하기 위한 데이터 준비
            data = patient_data[["측정날짜", metric]].rename(columns={"측정날짜": "ds", metric: "y"})

            # FBProphet 모델 생성 및 학습
            model = Prophet()
            model.fit(data)

            # 미래 데이터프레임 생성 및 예측
            future = model.make_future_dataframe(periods=30)  # 30일 예측
            forecast = model.predict(future)

            # 예측 결과 시각화
            fig1 = model.plot(forecast)
            fig2 = model.plot_components(forecast)
            
            st.pyplot(fig1)
            st.pyplot(fig2)
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
else:
    st.info("CSV 파일을 업로드하세요.")
