# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from utils.data_loader import load_excel_data
from utils.predictor import predict_district_prices

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.set_page_config(page_title="서울시 부동산 투자 추천", page_icon="🏠", layout="wide")

st.title("🏠 AI 기반 서울시 부동산 투자 추천 서비스")
st.markdown("""
**3대 알고리즘(Linear, RF, Prophet)**의 예측 결과를 비교 분석합니다.
각 모델의 **예상 수익률**과 **오차율**을 모두 확인하고, 가장 신뢰할 수 있는 모델을 참고하세요.
""")
st.divider()

# 사이드바
st.sidebar.header("⚙️ 설정 및 입력")
uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
months = st.sidebar.slider("미래 예측 기간 (개월)", min_value=1, max_value=60, value=12)

st.sidebar.divider()
st.sidebar.header("🔍 정렬 기준")
sort_option = st.sidebar.radio(
    "어떤 기준으로 추천할까요?",
    ("최적 모델 수익률 높은 순 (투자용)", "예상 미래 지수 높은 순 (자산가치용)")
)

# 세션 초기화
if 'results_df' not in st.session_state: st.session_state['results_df'] = None
if 'forecasts' not in st.session_state: st.session_state['forecasts'] = None
if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False

if uploaded_file is not None:
    df = load_excel_data(uploaded_file)

    if df is not None:
        st.success("✅ 데이터 로드 완료!")
        
        if st.button("🚀 AI 상세 분석 시작"):
            with st.spinner('3대 모델 전수 조사 및 오차율 검증 중...'):
                results_df, forecasts = predict_district_prices(df, months=months)
                st.session_state['results_df'] = results_df
                st.session_state['forecasts'] = forecasts
                st.session_state['data_loaded'] = True

        if st.session_state['data_loaded'] and st.session_state['results_df'] is not None:
            results_df = st.session_state['results_df']
            forecasts = st.session_state['forecasts']
            
            # 정렬 로직 (내부적으로 '최적 수익률' 컬럼을 사용)
            if sort_option == "최적 모델 수익률 높은 순 (투자용)":
                results_df = results_df.sort_values(by='최적 수익률', ascending=False)
                rank_title = "🔥 급상승 예상 지역 (Top 5)"
                color_map = 'Reds'
            else:
                # 자산가치 = 현재지수 * (1 + 최적수익률/100)
                results_df['예상 미래 지수'] = results_df['현재 지수'] * (1 + results_df['최적 수익률'] / 100)
                results_df = results_df.sort_values(by='예상 미래 지수', ascending=False)
                rank_title = "💎 미래 최고 부촌 예상 (Top 5)"
                color_map = 'Blues'
            
            excel_data = convert_df_to_excel(results_df)
            st.sidebar.divider()
            st.sidebar.download_button("📥 상세 결과 엑셀 다운로드", excel_data, 'seoul_housing_analysis.xlsx')

            st.subheader(f"🏆 {rank_title}")
            
            # [핵심 수정] 사용자가 요청한 대로 모든 컬럼 나열
            display_cols = [
                '자치구', '현재 지수',
                'Linear 수익률(%)', 'Linear 오차',
                'RF 수익률(%)', 'RF 오차',
                'Prophet 수익률(%)', 'Prophet 오차',
                '추천 모델'
            ]
            
            top5 = results_df.head(5)
            
            # 스타일링: 수익률이 높은 곳 강조, 오차율은 그대로 표시
            st.dataframe(
                top5[display_cols].style.background_gradient(subset=['Linear 수익률(%)', 'RF 수익률(%)', 'Prophet 수익률(%)'], cmap=color_map),
                use_container_width=True
            )
            
            with st.expander("📋 전체 자치구 상세 데이터 보기"):
                st.dataframe(results_df[display_cols])

            st.divider()

            st.subheader("📈 상세 분석 그래프")
            
            selected_district = st.selectbox("자치구 선택:", results_df['자치구'].unique(), index=0)
            
            row = results_df[results_df['자치구'] == selected_district].iloc[0]
            best_model_name = row['추천 모델']
            
            if 'RandomForest' in best_model_name: err_key = 'RF 오차'
            elif 'Linear' in best_model_name: err_key = 'Linear 오차'
            else: err_key = 'Prophet 오차'
            
            # 오차율 문자열('2.5%')에서 숫자만 추출하기 위해 로직 처리 필요없음 (이미 문자열)
            best_model_error = row[err_key]
            
            st.info(f"💡 **{selected_district}**의 분석 결과: **[{best_model_name}]**이(가) 오차율 **{best_model_error}**로 가장 신뢰할 수 있습니다.")
            
            data = forecasts[selected_district]
            history, prophet, linear, rf = data['history'], data['prophet'], data['linear'], data['rf']
            errors = data['errors']
            
            fig = go.Figure()
            
            # 실제 데이터
            fig.add_trace(go.Scatter(x=history['date'], y=history['price'], mode='lines', name='실제 가격', line=dict(color='#FF4B4B', width=2)))
            
            # Linear
            fig.add_trace(go.Scatter(x=linear['ds'], y=linear['yhat'], mode='lines', name=f'Linear (오차 {errors["Linear"]:.1f}%)', line=dict(color='#FFA500', width=2, dash='dot')))

            # Random Forest
            fig.add_trace(go.Scatter(x=rf['ds'], y=rf['yhat'], mode='lines', name=f'RF (오차 {errors["RandomForest"]:.1f}%)', line=dict(color='#9D00FF', width=2, dash='dash')))
            
            # Prophet
            fig.add_trace(go.Scatter(x=prophet['ds'], y=prophet['yhat'], mode='lines', name=f'Prophet (오차 {errors["Prophet"]:.1f}%)', line=dict(color='#00CC96', width=3)))
            
            # Prophet 범위 (복잡해지지 않게 Prophet만 대표로 표시하거나, 뺄 수도 있음. 여기선 둠)
            fig.add_trace(go.Scatter(x=prophet['ds'], y=prophet['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 204, 150, 0.05)', showlegend=False))
            
            fig.update_layout(title=f"{selected_district} : 3대 모델 전수 비교", xaxis_title="날짜", yaxis_title="지수", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("데이터 형식이 올바르지 않습니다.")
else:
    st.info("👈 엑셀 파일을 업로드해주세요.")
