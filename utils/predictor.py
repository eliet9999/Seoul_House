# utils/predictor.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st

def predict_district_prices(df, months=12):
    """
    3대 알고리즘에 대해 '최근 3년 교차 검증(Time-Series CV)'을 수행하여
    더 신뢰할 수 있는 평균 오차율(MAPE)을 계산합니다.
    """
    results = []
    forecasts = {}
    
    districts = df['district'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(districts)
    
    for i, district in enumerate(districts):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"⏳ 3년치 교차 검증 중...: {district} ({i+1}/{total})")
        
        # 해당 자치구 전체 데이터
        district_df = df[df['district'] == district].copy()
        
        # ---------------------------------------------------------
        # [Step 1] 시계열 교차 검증 (Cross Validation)
        # 최근 3년(36개월)에 대해 1년씩 잘라서 3번 테스트하고 평균을 냄
        # ---------------------------------------------------------
        
        # 검증 결과를 담을 리스트
        errors_p = []
        errors_l = []
        errors_rf = []
        
        # 데이터가 충분할 때만 3년치 검증 (최소 5년치 데이터 권장)
        if len(district_df) > 60:
            cv_years = 3 # 3년치 검증
        elif len(district_df) > 36:
            cv_years = 1 # 데이터 적으면 1년만 검증
        else:
            cv_years = 0 # 너무 적으면 검증 불가
            
        if cv_years > 0:
            # 3번 반복 (예: 2022 테스트 -> 2023 테스트 -> 2024 테스트)
            for k in range(cv_years, 0, -1):
                # 테스트 시점 설정 (뒤에서부터 12개월씩 끊음)
                # k=3: 3년 전까지 학습 -> 2년 전 데이터 맞추기
                # k=1: 1년 전까지 학습 -> 최근 1년 데이터 맞추기
                cut_idx = 12 * k 
                
                train_df = district_df.iloc[:-cut_idx] # 학습용
                if k == 1:
                    test_df = district_df.iloc[-12:]   # 마지막 1년
                else:
                    # k=2이면 뒤에서 24개월~12개월 사이 구간
                    test_df = district_df.iloc[-cut_idx : -(cut_idx-12)]
                
                # 데이터 준비
                X_train = train_df['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y_train = train_df['price'].values
                X_test = test_df['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y_test = test_df['price'].values
                
                # 1. Prophet 검증
                try:
                    p_train = train_df.rename(columns={'date': 'ds', 'price': 'y'})
                    m_p = Prophet(daily_seasonality=False, weekly_seasonality=False).fit(p_train)
                    p_pred = m_p.predict(test_df.rename(columns={'date': 'ds'}))['yhat'].values
                    errors_p.append(mean_absolute_percentage_error(y_test, p_pred) * 100)
                except:
                    errors_p.append(100.0) # 에러 시 벌칙 점수

                # 2. Linear 검증
                try:
                    m_l = LinearRegression().fit(X_train, y_train)
                    l_pred = m_l.predict(X_test)
                    errors_l.append(mean_absolute_percentage_error(y_test, l_pred) * 100)
                except:
                    errors_l.append(100.0)

                # 3. Random Forest 검증
                try:
                    m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
                    rf_pred = m_rf.predict(X_test)
                    errors_rf.append(mean_absolute_percentage_error(y_test, rf_pred) * 100)
                except:
                    errors_rf.append(100.0)
            
            # [최종 오차율] 3번 시험 본 것의 평균 점수
            avg_error_p = np.mean(errors_p)
            avg_error_l = np.mean(errors_l)
            avg_error_rf = np.mean(errors_rf)
            
        else:
            # 데이터 부족 시 기본값
            avg_error_p, avg_error_l, avg_error_rf = 99.9, 99.9, 99.9

        # ---------------------------------------------------------
        # [Step 2] 최종 미래 예측 (전체 데이터 사용)
        # ---------------------------------------------------------
        
        # 공통 데이터
        district_df['date_ordinal'] = district_df['date'].map(pd.Timestamp.toordinal)
        X_all = district_df[['date_ordinal']]
        y_all = district_df['price']
        
        # Prophet 학습
        prophet_final_df = district_df.rename(columns={'date': 'ds', 'price': 'y'})
        m_prophet = Prophet(daily_seasonality=False, weekly_seasonality=False)
        m_prophet.fit(prophet_final_df)
        future_prophet = m_prophet.make_future_dataframe(periods=months, freq='MS')
        fc_prophet = m_prophet.predict(future_prophet)
        
        future_dates = fc_prophet['ds']
        future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        
        # Linear 학습
        m_linear = LinearRegression()
        m_linear.fit(X_all, y_all)
        fc_linear = m_linear.predict(future_dates_ordinal)
        
        # Random Forest 학습
        m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        m_rf.fit(X_all, y_all)
        fc_rf = m_rf.predict(future_dates_ordinal)
        
        # --- 결과 저장 ---
        current_price = district_df['price'].iloc[-1]
        future_price_p = fc_prophet['yhat'].iloc[-1]
        return_p = (future_price_p - current_price) / current_price * 100
        
        best_error = min(avg_error_p, avg_error_l, avg_error_rf)
        if best_error == avg_error_p: best_model = "Prophet"
        elif best_error == avg_error_l: best_model = "Linear"
        else: best_model = "RandomForest"

        results.append({
            '자치구': district,
            '현재 지수': round(current_price, 2),
            '예상 변화율(%)': round(return_p, 2),
            '추천 모델': best_model,
            'Prophet 오차': f"{avg_error_p:.2f}%",
            'Linear 오차': f"{avg_error_l:.2f}%",
            'RandomForest 오차': f"{avg_error_rf:.2f}%"
        })
        
        forecasts[district] = {
            'history': district_df,
            'prophet': fc_prophet,
            'linear': pd.DataFrame({'ds': future_dates, 'yhat': fc_linear}),
            'rf': pd.DataFrame({'ds': future_dates, 'yhat': fc_rf}),
            'errors': {'Prophet': avg_error_p, 'Linear': avg_error_l, 'RandomForest': avg_error_rf}
        }

    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results).sort_values(by='예상 변화율(%)', ascending=False)
    
    return results_df, forecasts

