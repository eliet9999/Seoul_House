# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from utils.data_loader import load_excel_data
from utils.predictor import predict_district_prices

# 엑셀 다운로드 함수
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.set_page_config(page_title="서울시 부동산 투자 추천", page_icon="🏠", layout="wide")

st.title("🏠 AI 기반 서울시 부동산 투자 추천 서비스")
st.markdown("""
**3대 알고리즘(Prophet, Linear, RF)**을 통해 미래 가치를 예측합니다.
**Prophet 예상 수익률(성장성)**과 **예상 미래 지수(자산 가치)** 두 가지 관점으로 분석해 보세요.
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
sort_option = st.sidebar.radio(
    "어떤 기준으로 추천할까요?",
    ("Prophet 예상 수익률 높은 순 (투자용)", "예상 미래 지수 높은 순 (자산가치용)")
)

# -------------------------------------------------------------------------
# [핵심 수정] 세션 상태 초기화 (결과를 담을 그릇 만들기)
# -------------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None
if 'forecasts' not in st.session_state:
    st.session_state['forecasts'] = None
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# -------------------------------------------------------------------------
# 메인 로직
# -------------------------------------------------------------------------
if uploaded_file is not None:
    # 파일이 바뀌면 데이터 다시 로드
    df = load_excel_data(uploaded_file)

    if df is not None:
        st.success("✅ 데이터 로드 완료!")
        
        # 분석 버튼 클릭 시 실행
        if st.button("🚀 AI 분석 시작"):
            with st.spinner('3년치 교차 검증 및 미래 예측 분석 중... (시간이 조금 걸립니다)'):
                # 분석 수행
                results_df, forecasts = predict_district_prices(df, months=months)
                
                # [중요] 결과를 세션 저장소에 '영구 저장'
                st.session_state['results_df'] = results_df
                st.session_state['forecasts'] = forecasts
                st.session_state['data_loaded'] = True # 분석 완료 깃발 꽂기

        # ----------------------------------------------------------------
        # [수정됨] 버튼 안 눌러도, 저장된 결과가 있으면 화면에 표시!
        # ----------------------------------------------------------------
        if st.session_state['data_loaded'] and st.session_state['results_df'] is not None:
            
            # 저장된 데이터 꺼내오기
            results_df = st.session_state['results_df']
            forecasts = st.session_state['forecasts']
            
            # 정렬 로직 적용
            if sort_option == "Prophet 예상 수익률 높은 순 (투자용)":
                results_df = results_df.sort_values(by='Prophet 예상 수익률(%)', ascending=False)
                rank_title = "🔥 급상승 예상 지역 (수익률 Top 5)"
                color_map = 'Reds'
            else:
                results_df['예상 미래 지수'] = results_df['현재 지수'] * (1 + results_df['Prophet 예상 수익률(%)'] / 100)
                results_df = results_df.sort_values(by='예상 미래 지수', ascending=False)
                rank_title = "💎 미래 최고 부촌 예상 (지수 Top 5)"
                color_map = 'Blues'
            
            # 엑셀 다운로드
            excel_data = convert_df_to_excel(results_df)
            st.sidebar.divider()
            st.sidebar.download_button(
                label="📥 분석 결과 엑셀 다운로드",
                data=excel_data,
                file_name='seoul_housing_analysis.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            # 결과 표 표시
            st.subheader(f"🏆 {rank_title}")
            display_cols = ['자치구', '현재 지수', 'Prophet 예상 수익률(%)', '추천 모델', 'Prophet 오차', 'Linear 오차', 'RandomForest 오차']
            top5 = results_df.head(5)
            
            st.dataframe(
                top5[display_cols].style.background_gradient(subset=['Prophet 예상 수익률(%)'], cmap=color_map),
                use_container_width=True
            )
            
            with st.expander("📋 전체 순위 보기"):
                st.dataframe(results_df[display_cols])

            st.divider()

            # 상세 그래프
            st.subheader("📈 상세 분석 그래프")
            
            # 1등 지역 기본값 설정
            top_district = top5.iloc[0]['자치구']
            
            # [수정] selectbox를 바꿔도 이 안쪽 코드가 실행되므로 데이터가 유지됨
            selected_district = st.selectbox("자치구 선택:", results_df['자치구'].unique(), index=0)
            
            row = results_df[results_df['자치구'] == selected_district].iloc[0]
            best_model_name = row['추천 모델']
            
            if 'RandomForest' in best_model_name:
                err_key = 'RandomForest 오차'
            elif 'Linear' in best_model_name:
                err_key = 'Linear 오차'
            else:
                err_key = 'Prophet 오차'
            
            best_model_error = row[err_key]
            
            st.info(f"💡 **{selected_district}**의 분석 결과: **[{best_model_name}]** 모델이 가장 정확합니다. (오차율: {best_model_error})")
            
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
