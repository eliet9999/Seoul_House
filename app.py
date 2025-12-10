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
**3대 알고리즘(Linear, RF, Prophet)**의 시나리오별 미래 가치를 분석합니다.
좌측 사이드바에서 **정렬 기준**을 변경하여 모델별 예상 순위를 비교해 보세요.
""")
st.divider()

# -------------------------------------------------------------------------
# [사이드바] 설정
# -------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정 및 입력")
uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
months = st.sidebar.slider("미래 예측 기간 (개월)", min_value=1, max_value=60, value=12)

st.sidebar.divider()
st.sidebar.header("🔍 정렬 기준 (Ranking)")

# [핵심 수정] 정렬 옵션을 세분화하여 추가
sort_option = st.sidebar.radio(
    "어떤 기준으로 순위를 볼까요?",
    (
        "🔥 통합 추천: 급상승 예상 (수익률 순)",
        "💎 통합 추천: 미래 부촌 (지수 순)",
        "📏 Linear 기준: 미래 부촌 (지수 순)",
        "🌲 RF 기준: 미래 부촌 (지수 순)",
        "🔮 Prophet 기준: 미래 부촌 (지수 순)"
    )
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
            with st.spinner('3대 모델 전수 조사 및 시나리오 분석 중...'):
                results_df, forecasts = predict_district_prices(df, months=months)
                st.session_state['results_df'] = results_df
                st.session_state['forecasts'] = forecasts
                st.session_state['data_loaded'] = True

        if st.session_state['data_loaded'] and st.session_state['results_df'] is not None:
            results_df = st.session_state['results_df'].copy() # 원본 보존을 위해 copy
            forecasts = st.session_state['forecasts']
            
            # ----------------------------------------------------------------
            # [로직 수정] 선택한 기준에 따라 정렬 및 미래 지수 계산
            # ----------------------------------------------------------------
            if "🔥 통합 추천" in sort_option:
                # 최적 모델 수익률 기준
                results_df = results_df.sort_values(by='최적 수익률', ascending=False)
                rank_title = "🔥 급상승 예상 지역 (통합 Top 5)"
                color_map = 'Reds'
                highlight_col = '최적 수익률' # 수익률 강조
                
            else:
                # 자산 가치(지수) 기준 정렬 로직
                rank_title = f"🏆 {sort_option.split(':')[0]} Top 5"
                color_map = 'Blues'
                
                if "💎 통합 추천" in sort_option:
                    target_return_col = '최적 수익률'
                elif "Linear" in sort_option:
                    target_return_col = 'Linear 수익률(%)'
                elif "RF" in sort_option:
                    target_return_col = 'RF 수익률(%)'
                elif "Prophet" in sort_option:
                    target_return_col = 'Prophet 수익률(%)'
                
                # 선택된 모델의 수익률을 기반으로 '예상 미래 지수' 계산
                results_df['시나리오별 미래 지수'] = results_df['현재 지수'] * (1 + results_df[target_return_col] / 100)
                results_df = results_df.sort_values(by='시나리오별 미래 지수', ascending=False)
                highlight_col = target_return_col

            # ----------------------------------------------------------------
            
            excel_data = convert_df_to_excel(results_df)
            st.sidebar.divider()
            st.sidebar.download_button("📥 현재 결과 엑셀 다운로드", excel_data, 'seoul_housing_analysis.xlsx')

            st.subheader(rank_title)
            
            # 표시할 컬럼 정의
            display_cols = [
                '자치구', '현재 지수',
                'Linear 수익률(%)', 'Linear 오차',
                'RF 수익률(%)', 'RF 오차',
                'Prophet 수익률(%)', 'Prophet 오차',
                '추천 모델'
            ]
            
            # 미래 지수 모드일 경우, 계산된 미래 지수도 보여주면 좋음
            if "통합 추천" not in sort_option or "미래 부촌" in sort_option:
                display_cols.insert(2, '시나리오별 미래 지수')

            top5 = results_df.head(5)
            
            # 스타일링: 선택된 기준 모델의 수익률 컬럼을 강조
            st.dataframe(
                top5[display_cols].style.background_gradient(subset=[highlight_col], cmap=color_map),
                use_container_width=True
            )
            
            with st.expander("📋 전체 자치구 순위 보기"):
                st.dataframe(results_df[display_cols])

            st.divider()

            st.subheader("📈 상세 분석 그래프")
            
            # [중요] 정렬된 순서대로 Selectbox 목록이 나옴 -> 1등부터 차례로 보기 편함
            selected_district = st.selectbox("자치구 선택 (위 순위대로 정렬됨):", results_df['자치구'].unique(), index=0)
            
            row = results_df[results_df['자치구'] == selected_district].iloc[0]
            best_model_name = row['추천 모델']
            
            if 'RandomForest' in best_model_name: err_key = 'RF 오차'
            elif 'Linear' in best_model_name: err_key = 'Linear 오차'
            else: err_key = 'Prophet 오차'
            
            best_model_error = row[err_key]
            
            st.info(f"💡 **{selected_district}**의 분석 결과: 오차율 **{best_model_error}**인 **[{best_model_name}]** 모델이 가장 신뢰도가 높습니다.")
            
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
            
            fig.update_layout(title=f"{selected_district} : 3대 모델 전수 비교", xaxis_title="날짜", yaxis_title="지수", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("데이터 형식이 올바르지 않습니다.")
else:
    st.info("👈 엑셀 파일을 업로드해주세요.")
