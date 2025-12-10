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
**3대 알고리즘(Linear, RF, Prophet)**의 예측 결과를 시나리오별로 비교합니다.
분석 후 **순위 결정 모델**을 변경하여 모델별 자치구 순위 변동을 확인하세요.
""")
st.divider()

# -------------------------------------------------------------------------
# [사이드바] 기본 설정
# -------------------------------------------------------------------------
st.sidebar.header("⚙️ 설정 및 입력")
uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드 (.xlsx)", type=["xlsx"])
months = st.sidebar.slider("미래 예측 기간 (개월)", min_value=1, max_value=60, value=12)

st.sidebar.divider()
st.sidebar.header("🎯 분석 목표 (View)")
view_option = st.sidebar.radio(
    "무엇을 중점으로 볼까요?",
    ("예상 수익률 높은 순 (투자 가치)", "예상 미래 지수 높은 순 (자산 가치)")
)

# 세션 초기화
if 'results_df' not in st.session_state: st.session_state['results_df'] = None
if 'forecasts' not in st.session_state: st.session_state['forecasts'] = None
if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False

if uploaded_file is not None:
    df = load_excel_data(uploaded_file)

    if df is not None:
        st.success("✅ 데이터 로드 완료!")
        
        # 분석 버튼
        if st.button("🚀 AI 분석 시작"):
            with st.spinner('3대 모델 전수 조사 및 교차 검증 중...'):
                results_df, forecasts = predict_district_prices(df, months=months)
                st.session_state['results_df'] = results_df
                st.session_state['forecasts'] = forecasts
                st.session_state['data_loaded'] = True

        # 분석 완료 후 화면 표시
        if st.session_state['data_loaded'] and st.session_state['results_df'] is not None:
            results_df = st.session_state['results_df'].copy()
            forecasts = st.session_state['forecasts']
            
            st.divider()
            
            # ----------------------------------------------------------------
            # [메인] 순위 결정 모델 선택
            # ----------------------------------------------------------------
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("### 📉 순위 결정 모델")
                ranking_model = st.selectbox(
                    "어떤 모델을 기준으로 등수를 매길까요?",
                    (
                        "🏆 AI 통합 추천 (최적 모델)",
                        "📏 Linear Regression (선형회귀)",
                        "🔮 Prophet (프로펫)",
                        "🌲 Random Forest (랜덤포레스트)"
                    )
                )

            # ----------------------------------------------------------------
            # [로직] 선택한 모델에 따라 정렬 컬럼 결정
            # ----------------------------------------------------------------
            if "AI 통합 추천" in ranking_model:
                target_return_col = '최적 수익률' # 이 컬럼이 표에 있어야 함!
                display_msg = "오차율이 가장 낮은 모델을 자동으로 반영한 순위입니다."
            elif "Linear" in ranking_model:
                target_return_col = 'Linear 수익률(%)'
                display_msg = "상승/하락 추세선을 기준으로 한 순위입니다."
            elif "Prophet" in ranking_model:
                target_return_col = 'Prophet 수익률(%)'
                display_msg = "계절성과 트렌드를 반영한 Prophet 모델 기준 순위입니다."
            elif "Random Forest" in ranking_model:
                target_return_col = 'RF 수익률(%)'
                display_msg = "최근 패턴을 보수적으로 반영한 Random Forest 기준 순위입니다."

            # ----------------------------------------------------------------
            # [로직] 정렬 수행
            # ----------------------------------------------------------------
            if "투자 가치" in view_option:
                results_df = results_df.sort_values(by=target_return_col, ascending=False)
                rank_title = f"{ranking_model.split('(')[0]} 기준 Top 5 (수익률)"
                color_map = 'Reds'
            else:
                results_df['시나리오별 미래 지수'] = results_df['현재 지수'] * (1 + results_df[target_return_col] / 100)
                results_df = results_df.sort_values(by='시나리오별 미래 지수', ascending=False)
                rank_title = f"{ranking_model.split('(')[0]} 기준 Top 5 (지수)"
                color_map = 'Blues'
            
            with col2:
                st.info(f"💡 **{display_msg}**")

            # ----------------------------------------------------------------
            # 결과 표 출력
            # ----------------------------------------------------------------
            st.subheader(f"📊 {rank_title}")
            
            # [수정됨] 기본 표시 컬럼
            display_cols = [
                '자치구', '현재 지수',
                'Linear 수익률(%)', 'Linear 오차',
                'RF 수익률(%)', 'RF 오차',
                'Prophet 수익률(%)', 'Prophet 오차',
                '추천 모델'
            ]
            
            # [핵심 수정] 정렬 기준이 되는 컬럼(예: 최적 수익률)이 리스트에 없으면 강제로 추가
            # 이렇게 해야 '없는 컬럼을 색칠해라'라는 에러가 안 남
            if target_return_col not in display_cols:
                # '현재 지수' 바로 뒤에 삽입해서 잘 보이게 함
                display_cols.insert(2, target_return_col)

            # 미래 지수 보기 모드면 컬럼 추가
            if "자산 가치" in view_option:
                # 중복 추가 방지
                if '시나리오별 미래 지수' not in display_cols:
                    display_cols.insert(2, '시나리오별 미래 지수')

            top5 = results_df.head(5)
            
            # 데이터프레임 표시 (이제 에러 안 남)
            st.dataframe(
                top5[display_cols].style.background_gradient(subset=[target_return_col], cmap=color_map),
                use_container_width=True
            )
            
            with st.expander("📋 전체 자치구 순위 보기 (엑셀 다운로드)"):
                st.dataframe(results_df[display_cols])
                excel_data = convert_df_to_excel(results_df)
                st.download_button("📥 전체 결과 엑셀 다운로드", excel_data, 'seoul_housing_analysis.xlsx')

            st.divider()

            # ----------------------------------------------------------------
            # 상세 그래프
            # ----------------------------------------------------------------
            st.subheader("📈 상세 시각화 및 모델 비교")
            
            selected_district = st.selectbox(
                "확인할 자치구를 선택하세요 (위 순위대로 정렬됨):", 
                results_df['자치구'].unique(), 
                index=0
            )
            
            row = results_df[results_df['자치구'] == selected_district].iloc[0]
            
            # 선택된 모델의 수익률 표시
            if "AI" in ranking_model:
                # AI 통합 추천일 때는 '최적 수익률' 값을 보여줌
                model_name = row['추천 모델']
                val = row['최적 수익률']
            elif "Linear" in ranking_model:
                model_name = "Linear Regression"
                val = row['Linear 수익률(%)']
            elif "Prophet" in ranking_model:
                model_name = "Prophet"
                val = row['Prophet 수익률(%)']
            else:
                model_name = "Random Forest"
                val = row['RF 수익률(%)']

            st.markdown(f"""
            ### 📌 {selected_district} 분석 요약
            * **[{ranking_model.split('(')[0]}]** 기준 예상 수익률: **{val:.2f}%**
            * (참고: 이 지역 최적 모델은 **{row['추천 모델']}** 입니다.)
            """)
            
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
