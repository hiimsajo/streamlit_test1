import streamlit as st
import pandas as pd
from fbprophet import Prophet
import plotly.graph_objs as go
from streamlit_option_menu import option_menu

# 사이드바 메뉴
with st.sidebar:
    selected = option_menu(
        "메뉴", ["데이터 업로드 및 선택", "예측 결과"],
        icons=["cloud-upload", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

# 데이터 업로드 및 선택
if selected == "데이터 업로드 및 선택":
    st.title("데이터 업로드 및 선택")
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        
        option = st.selectbox("환자번호 또는 환자이름으로 선택하세요", ["환자번호", "환자이름"])
        if option == "환자번호":
            patients = data["환자번호"].unique()
        else:
            patients = data["환자이름"].unique()
        
        selected_patient = st.selectbox(f"{option}을 선택하세요", patients)

        # 선택한 환자의 데이터 필터링
        if option == "환자번호":
            patient_data = data[data["환자번호"] == selected_patient]
        else:
            patient_data = data[data["환자이름"] == selected_patient]
        
        st.write(f"선택된 환자 데이터: {selected_patient}")
        st.write(patient_data)

        # 환자 데이터 저장
        st.session_state['patient_data'] = patient_data

# 예측 결과
elif selected == "예측 결과":
    st.title("예측 결과")
    
    if 'patient_data' in st.session_state:
        patient_data = st.session_state['patient_data']
        metrics = ["수축기혈압", "이완기혈압", "맥박", "혈당", "체온", "호흡", "체중"]
        
        for metric in metrics:
            st.subheader(f"{metric} 예측")
            data = patient_data[["측정날짜", metric]].rename(columns={"측정날짜": "ds", metric: "y"})
            data['ds'] = pd.to_datetime(data['ds'])
            
            # FBProphet 모델 생성 및 학습
            model = Prophet()
            model.fit(data)

            # 미래 데이터프레임 생성 및 예측
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            # 예측 결과 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'], mode='lines',
                fill='tonexty', line=dict(width=0), showlegend=False, name='Uncertainty Interval'))
            fig.update_layout(title=f"{metric} 예측", xaxis_title="날짜", yaxis_title=metric)
            st.plotly_chart(fig)
    else:
        st.write("먼저 데이터를 업로드하고 환자를 선택하세요.")
