# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io # [추가] 엑셀 파일을 메모리에서 다루기 위한 도구
from utils.data_loader import load_excel_data
from utils.predictor import predict_district_prices

# 엑셀 다운로드용 함수 (메모리에 파일을 저장함)
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.set_page_config(page_title="서울시 부동산 투자 추천", page_icon="🏠", layout="wide")

st.title("🏠 AI 기반 서울시 부동산 투자 추천 서비스")
st.markdown("""
**3대 알고리즘(Prophet, Linear, RF)**을 비교 분석합니다.
각 모델의 **평균 오차율(MAPE)**을 계산하여 가장 신뢰할 수 있는 예측을 추천합니다.
*(오차율이 낮을수록 정확한 모델입니다)*
""")
st.divider()

st.sidebar.header("⚙️ 설정 및 입력")
uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
months = st.sidebar.slider("미래 예측 기간 (개월)", min_value=1, max_value=60, value=12)

if uploaded_file is not None:
    df = load_excel_data(uploaded_file)

    if df is not None:
        st.success("✅ 데이터 로드 완료!")
        
        if st.button("🚀 정밀 분석 및 검증 시작"):
            with st.spinner('최근 1년 데이터로 오차율 테스트 중...'):
                results_df, forecasts = predict_district_prices(df, months=months)
            
            # ----------------------------------------------------------------
            # [기능 추가] 엑셀 다운로드 버튼
            # ----------------------------------------------------------------
            excel_data = convert_df_to_excel(results_df)
            
            st.sidebar.divider()
            st.sidebar.header("💾 결과 저장")
            st.sidebar.download_button(
                label="📥 분석 결과 엑셀 다운로드",
                data=excel_data,
                file_name='seoul_housing_analysis.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            # ----------------------------------------------------------------

            st.subheader(f"🏆 투자 유망 Top 5 지역 (수익률 순)")
            
            display_cols = ['자치구', '현재 지수', '예상 수익률(%)', '추천 모델', 'Prophet 오차', 'Linear 오차', 'RandomForest 오차']
            top5 = results_df[display_cols].head(5)
            
            st.dataframe(
                top5.style.background_gradient(subset=['예상 수익률(%)'], cmap='summer'),
                use_container_width=True
            )
            
            with st.expander("📋 전체 지역 오차율 및 상세 데이터 보기"):
                st.dataframe(results_df[display_cols])

            st.divider()

            st.subheader("📈 알고리즘 비교 및 오차 검증")
            
            top_district = top5.iloc[0]['자치구']
            selected_district = st.selectbox("자치구 선택:", results_df['자치구'], index=0)
            
            row = results_df[results_df['자치구'] == selected_district].iloc[0]
            best_model_name = row['추천 모델']
            best_model_error = row[f'{best_model_name} 오차']
            
            st.info(f"💡 **{selected_district}**의 경우, **[{best_model_name}]** 모델의 오차율이 **{best_model_error}**로 가장 낮아 신뢰도가 높습니다.")
            
            data = forecasts[selected_district]
            history = data['history']
            prophet = data['prophet']
            linear = data['linear']
            rf = data['rf']
            errors = data['errors']
            
            fig = go.Figure()
            
            # 1. 실제 데이터
            fig.add_trace(go.Scatter(
                x=history['date'], y=history['price'],
                mode='lines', name='실제 가격',
                line=dict(color='#FF4B4B', width=2)
            ))
            
            # 2. Prophet
            fig.add_trace(go.Scatter(
                x=prophet['ds'], y=prophet['yhat'],
                mode='lines', name=f'Prophet (오차 {errors["Prophet"]:.1f}%)',
                line=dict(color='#00CC96', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=prophet['ds'], y=prophet['yhat_upper'],
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=prophet['ds'], y=prophet['yhat_lower'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(0, 204, 150, 0.1)', name='AI 범위'
            ))
            
            # 3. Linear
            fig.add_trace(go.Scatter(
                x=linear['ds'], y=linear['yhat'],
                mode='lines', name=f'Linear (오차 {errors["Linear"]:.1f}%)',
                line=dict(color='#FFA500', width=2, dash='dot')
            ))
            
            # 4. Random Forest
            fig.add_trace(go.Scatter(
                x=rf['ds'], y=rf['yhat'],
                mode='lines', name=f'Random Forest (오차 {errors["RandomForest"]:.1f}%)',
                line=dict(color='#9D00FF', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"{selected_district} : 모델별 예측 비교",
                xaxis_title="날짜", yaxis_title="지수",
                hovermode="x unified",
                xaxis=dict(tickmode='linear', dtick="M12", tickformat="%Y년"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("데이터 형식이 올바르지 않습니다.")
else:
    st.info("👈 엑셀 파일을 업로드해주세요.")