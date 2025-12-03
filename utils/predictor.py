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
    3대 알고리즘의 '평균 오차율(MAPE)'을 평가하고 미래를 예측합니다.
    (오차율은 낮을수록 좋습니다.)
    """
    results = []
    forecasts = {}
    
    districts = df['district'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(districts)
    
    for i, district in enumerate(districts):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"⏳ 분석 및 오차율 검증 중: {district} ({i+1}/{total})")
        
        # 전체 데이터
        district_df = df[df['district'] == district].copy()
        
        # ---------------------------------------------------------
        # [Step 1] 모델 성능 평가 (Backtesting)
        # ---------------------------------------------------------
        # 기본값 (데이터 부족 시)
        error_p, error_l, error_rf = 99.9, 99.9, 99.9 
        
        if len(district_df) > 24:
            train_df = district_df.iloc[:-12]
            test_df = district_df.iloc[-12:]
            
            X_train = train_df['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y_train = train_df['price'].values
            X_test = test_df['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y_test = test_df['price'].values
            
            # 1. Prophet 평가
            p_train = train_df.rename(columns={'date': 'ds', 'price': 'y'})
            m_p_eval = Prophet(daily_seasonality=False, weekly_seasonality=False).fit(p_train)
            p_pred = m_p_eval.predict(test_df.rename(columns={'date': 'ds'}))['yhat'].values
            
            # 2. Linear 평가
            m_l_eval = LinearRegression().fit(X_train, y_train)
            l_pred = m_l_eval.predict(X_test)
            
            # 3. Random Forest 평가
            m_rf_eval = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            rf_pred = m_rf_eval.predict(X_test)
            
            # [오차율 계산] MAPE
            error_p = mean_absolute_percentage_error(y_test, p_pred) * 100
            error_l = mean_absolute_percentage_error(y_test, l_pred) * 100
            error_rf = mean_absolute_percentage_error(y_test, rf_pred) * 100

        # ---------------------------------------------------------
        # [Step 2] 최종 미래 예측
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
        
        best_error = min(error_p, error_l, error_rf)
        if best_error == error_p: best_model = "Prophet"
        elif best_error == error_l: best_model = "Linear"
        else: best_model = "RandomForest"

        results.append({
            '자치구': district,
            '현재 지수': round(current_price, 2),
            '예상 수익률(%)': round(return_p, 2),
            '추천 모델': best_model,
            'Prophet 오차': f"{error_p:.2f}%",
            'Linear 오차': f"{error_l:.2f}%",
            'RandomForest 오차': f"{error_rf:.2f}%" # [수정됨] 이름 통일
        })
        
        forecasts[district] = {
            'history': district_df,
            'prophet': fc_prophet,
            'linear': pd.DataFrame({'ds': future_dates, 'yhat': fc_linear}),
            'rf': pd.DataFrame({'ds': future_dates, 'yhat': fc_rf}),
            'errors': {'Prophet': error_p, 'Linear': error_l, 'RandomForest': error_rf}
        }

    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results).sort_values(by='예상 수익률(%)', ascending=False)
    
    return results_df, forecasts