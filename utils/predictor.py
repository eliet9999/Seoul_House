# utils/predictor.py
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import traceback

def predict_district_prices(df, months=12):
    """
    3대 알고리즘의 예측 결과(수익률)와 오차율을 모두 계산하여 반환합니다.
    """
    results = []
    forecasts = {}
    
    districts = df['district'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(districts)
    
    for i, district in enumerate(districts):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"⏳ 3대 모델 전수 분석 중...: {district} ({i+1}/{total})")
        
        district_df = df[df['district'] == district].copy()
        
        # ---------------------------------------------------------
        # [Step 1] 시계열 교차 검증 (오차율 계산)
        # ---------------------------------------------------------
        errors_p, errors_l, errors_rf = [], [], []
        
        if len(district_df) < 12:
            print(f"⚠️ [데이터 부족] {district}")
            continue

        if len(district_df) > 60: cv_years = 3
        elif len(district_df) > 36: cv_years = 1
        else: cv_years = 0
            
        if cv_years > 0:
            for k in range(cv_years, 0, -1):
                cut_idx = 12 * k 
                train_df = district_df.iloc[:-cut_idx]
                if k == 1: test_df = district_df.iloc[-12:]
                else: test_df = district_df.iloc[-cut_idx : -(cut_idx-12)]
                
                if len(train_df) < 2 or len(test_df) < 1: continue

                X_train = train_df['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y_train = train_df['price'].values
                X_test = test_df['date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y_test = test_df['price'].values
                
                # 1. Prophet
                try:
                    p_train = train_df.rename(columns={'date': 'ds', 'price': 'y'})
                    m_p = Prophet(daily_seasonality=False, weekly_seasonality=False).fit(p_train)
                    p_pred = m_p.predict(test_df.rename(columns={'date': 'ds'}))['yhat'].values
                    errors_p.append(mean_absolute_percentage_error(y_test, p_pred) * 100)
                except: errors_p.append(100.0)

                # 2. Linear
                try:
                    m_l = LinearRegression().fit(X_train, y_train)
                    l_pred = m_l.predict(X_test)
                    errors_l.append(mean_absolute_percentage_error(y_test, l_pred) * 100)
                except: errors_l.append(100.0)

                # 3. RF
                try:
                    m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
                    rf_pred = m_rf.predict(X_test)
                    errors_rf.append(mean_absolute_percentage_error(y_test, rf_pred) * 100)
                except: errors_rf.append(100.0)
            
            avg_error_p = np.mean(errors_p) if errors_p else 99.9
            avg_error_l = np.mean(errors_l) if errors_l else 99.9
            avg_error_rf = np.mean(errors_rf) if errors_rf else 99.9
        else:
            avg_error_p, avg_error_l, avg_error_rf = 99.9, 99.9, 99.9

        # ---------------------------------------------------------
        # [Step 2] 최종 미래 예측 (모든 모델 수행)
        # ---------------------------------------------------------
        try:
            district_df['date_ordinal'] = district_df['date'].map(pd.Timestamp.toordinal)
            X_all = district_df[['date_ordinal']]
            y_all = district_df['price']
            
            # Prophet
            prophet_final_df = district_df.rename(columns={'date': 'ds', 'price': 'y'})
            m_prophet = Prophet(daily_seasonality=False, weekly_seasonality=False)
            m_prophet.fit(prophet_final_df)
            future_prophet = m_prophet.make_future_dataframe(periods=months, freq='MS')
            fc_prophet = m_prophet.predict(future_prophet)
            future_dates = fc_prophet['ds']
            future_dates_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            
            # Linear
            m_linear = LinearRegression()
            m_linear.fit(X_all, y_all)
            fc_linear = m_linear.predict(future_dates_ordinal)
            
            # RF
            m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            m_rf.fit(X_all, y_all)
            fc_rf = m_rf.predict(future_dates_ordinal)
            
            # ---------------------------------------------------------
            # [Step 3] 결과 정리 (모든 값 저장)
            # ---------------------------------------------------------
            current_price = district_df['price'].iloc[-1]
            
            # 1. 미래 가격 추출
            future_p = fc_prophet['yhat'].iloc[-1]
            future_l = fc_linear[-1]
            future_rf = fc_rf[-1]
            
            # 2. 수익률 각각 계산
            return_p = (future_p - current_price) / current_price * 100
            return_l = (future_l - current_price) / current_price * 100
            return_rf = (future_rf - current_price) / current_price * 100
            
            # 3. 최적 모델 판별
            best_error = min(avg_error_p, avg_error_l, avg_error_rf)
            
            if best_error == avg_error_p:
                best_model = "Prophet"
                best_return = return_p
            elif best_error == avg_error_l:
                best_model = "Linear"
                best_return = return_l
            else:
                best_model = "RandomForest"
                best_return = return_rf

            # 4. 결과 저장 (모든 정보 포함)
            results.append({
                '자치구': district,
                '현재 지수': round(current_price, 2),
                
                # 정렬을 위한 내부 점수 (화면엔 안 보여줘도 됨)
                '최적 수익률': best_return, 
                
                # Linear
                'Linear 수익률(%)': round(return_l, 2),
                'Linear 오차': f"{avg_error_l:.2f}%",
                
                # Random Forest
                'RF 수익률(%)': round(return_rf, 2),
                'RF 오차': f"{avg_error_rf:.2f}%",
                
                # Prophet
                'Prophet 수익률(%)': round(return_p, 2),
                'Prophet 오차': f"{avg_error_p:.2f}%",
                
                # 결론
                '추천 모델': best_model
            })
            
            forecasts[district] = {
                'history': district_df,
                'prophet': fc_prophet,
                'linear': pd.DataFrame({'ds': future_dates, 'yhat': fc_linear}),
                'rf': pd.DataFrame({'ds': future_dates, 'yhat': fc_rf}),
                'errors': {'Prophet': avg_error_p, 'Linear': avg_error_l, 'RandomForest': avg_error_rf}
            }
            
        except Exception as e:
            print(f"❌ {district} 에러: {e}")
            print(traceback.format_exc())
            continue

    progress_bar.empty()
    status_text.empty()
    # 정렬은 '최적 모델의 수익률' 기준으로 함 (그래야 Top 5가 의미 있음)
    results_df = pd.DataFrame(results).sort_values(by='최적 수익률', ascending=False)
    return results_df, forecasts
