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
    results = []
    forecasts = {}
    
    districts = df['district'].unique()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(districts)
    
    for i, district in enumerate(districts):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"⏳ 분석 중...: {district} ({i+1}/{total})")
        
        district_df = df[df['district'] == district].copy()
        
        # ---------------------------------------------------------
        # [Step 1] 시계열 교차 검증
        # ---------------------------------------------------------
        errors_p, errors_l, errors_rf = [], [], []
        
        if len(district_df) < 12:
            print(f"⚠️ [데이터 부족] {district}: 건너뜁니다.")
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
                
                try:
                    p_train = train_df.rename(columns={'date': 'ds', 'price': 'y'})
                    m_p = Prophet(daily_seasonality=False, weekly_seasonality=False).fit(p_train)
                    p_pred = m_p.predict(test_df.rename(columns={'date': 'ds'}))['yhat'].values
                    errors_p.append(mean_absolute_percentage_error(y_test, p_pred) * 100)
                except: errors_p.append(100.0)

                try:
                    m_l = LinearRegression().fit(X_train, y_train)
                    l_pred = m_l.predict(X_test)
                    errors_l.append(mean_absolute_percentage_error(y_test, l_pred) * 100)
                except: errors_l.append(100.0)

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
        # [Step 2] 최종 미래 예측
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
            
            # Linear & RF
            m_linear = LinearRegression()
            m_linear.fit(X_all, y_all)
            fc_linear = m_linear.predict(future_dates_ordinal)
            
            m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            m_rf.fit(X_all, y_all)
            fc_rf = m_rf.predict(future_dates_ordinal)
            
            # ---------------------------------------------------------
            # [Step 3] 결과 정리
            # ---------------------------------------------------------
            current_price = district_df['price'].iloc[-1]
            future_p = fc_prophet['yhat'].iloc[-1]
            future_l = fc_linear[-1]
            future_rf = fc_rf[-1]
            
            best_error = min(avg_error_p, avg_error_l, avg_error_rf)
            
            if best_error == avg_error_p:
                best_model = "Prophet"
                final_future_price = future_p
            elif best_error == avg_error_l:
                best_model = "Linear"
                final_future_price = future_l
            else:
                best_model = "RandomForest"
                final_future_price = future_rf

            final_return = (final_future_price - current_price) / current_price * 100

            results.append({
                '자치구': district,
                '현재 지수': round(current_price, 2),
                'Prophet 예상 수익률(%)': round(final_return, 2), # [중요] 이름 변경됨!
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
            
        except Exception as e:
            print(f"❌ [오류] {district}: {e}")
            print(traceback.format_exc())
            continue

    progress_bar.empty()
    status_text.empty()
    # 정렬 기준도 바뀐 이름으로 수정
    results_df = pd.DataFrame(results).sort_values(by='Prophet 예상 수익률(%)', ascending=False)
    return results_df, forecasts
