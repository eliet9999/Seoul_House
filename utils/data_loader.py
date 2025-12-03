import pandas as pd
import streamlit as st

def load_excel_data(uploaded_file):
    """
    가로형(Wide) 엑셀 데이터를 읽어 세로형(Long)으로 변환합니다.
    형식: 자치구별(2) 컬럼에 구 이름이 있고, 나머지 컬럼이 날짜('2014. 01' 등)인 경우
    """
    if uploaded_file is None:
        return None

    try:
        # 1. 엑셀 파일 읽기
        # header=0: 첫 번째 줄을 제목으로 읽음
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # 2. 불필요한 행 제거 (예: '아파트' 등이 적힌 줄이나 빈 줄)
        # '자치구별(2)' 컬럼에서 실제 구 이름이 아닌 것들('소계', '아파트' 등)을 걸러낼 수 있습니다.
        # 하지만 일단 데이터 구조를 먼저 바꾼 뒤 처리하는 게 안전합니다.

        # 컬럼명에 '자치구별(2)'가 있는지 확인 (이미지 기준)
        district_col = '자치구별(2)'
        if district_col not in df.columns:
            # 만약 엑셀에 '자치구별(2)'가 없다면, 두 번째 컬럼을 구 이름으로 가정
            district_col = df.columns[1] 

        # 3. 데이터 구조 변경 (Wide -> Long) aka 'Melt'
        # 식별자 변수(id_vars): 고정할 컬럼 (자치구)
        # 나머지 모든 컬럼은 날짜로 간주하여 행으로 내립니다.
        
        # '자치구별(1)' 같은 불필요한 컬럼은 미리 제거
        if '자치구별(1)' in df.columns:
            df = df.drop(columns=['자치구별(1)'])

        # Melt 실행
        df_melted = df.melt(id_vars=[district_col], var_name='date', value_name='price')

        # 4. 컬럼 이름 표준화
        df_melted = df_melted.rename(columns={district_col: 'district'})

        # 5. 데이터 정제 (Cleaning)
        
        # 5-1. 'district' 컬럼 정제: '소계', '아파트' 같은 행 제거 및 공백 제거
        df_melted['district'] = df_melted['district'].astype(str).str.strip()
        # '소계', '서울', '아파트', 'nan' 등이 포함된 행 제거
        exclude_keywords = ['소계', '서울', '아파트', 'nan', '전국']
        mask = df_melted['district'].apply(lambda x: not any(keyword in x for keyword in exclude_keywords))
        df_melted = df_melted[mask]

        # 5-2. 'date' 컬럼 정제 (문자열 '2014. 01' -> 날짜형 2014-01-01)
        # 엑셀에서 날짜가 문자열로 오거나, 이미 날짜형일 수도 있습니다.
        
        def parse_date(x):
            try:
                # 만약 이미 datetime 객체라면 그대로 반환
                if isinstance(x, pd.Timestamp):
                    return x
                # 문자열이라면 '2014. 01' 형식을 처리
                str_x = str(x).replace('.', '-').strip() # 2014. 01 -> 2014- 01
                return pd.to_datetime(str_x)
            except:
                return None

        df_melted['date'] = df_melted['date'].apply(parse_date)
        
        # 날짜 변환 실패(None)한 행 제거 (이상한 컬럼이 섞여있을 경우 대비)
        df_melted = df_melted.dropna(subset=['date'])

        # 5-3. 'price' 컬럼 정제 (숫자로 변환)
        df_melted['price'] = pd.to_numeric(df_melted['price'], errors='coerce')
        df_melted = df_melted.dropna(subset=['price']) # 숫자가 아닌 값(결측치) 제거

        # 6. 최종 컬럼 확인 및 정렬
        final_df = df_melted[['date', 'district', 'price']].sort_values(by=['district', 'date']).reset_index(drop=True)

        return final_df

    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
        return None