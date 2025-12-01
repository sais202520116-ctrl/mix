import streamlit as st
import pandas as pd
import numpy as np

# Streamlit 앱의 타이틀 설정
st.title('🏋️‍♂️ 피트니스 데이터 상관관계 분석 앱')
st.subheader('업로드된 데이터 파일: fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv')

# 데이터 파일 이름
FILE_NAME = "fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data(file_path):
    """CSV 파일을 로드하고 분석을 위해 전처리합니다."""
    try:
        # 데이터 로드 (쉼표를 구분자로 사용)
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 숫자형 데이터만 선택
        # 한글 컬럼명이 있어, 숫자형으로 변환 가능한 컬럼을 필터링합니다.
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 모든 숫자형 데이터에 대해 상관관계 계산
        correlation_matrix = numeric_df.corr()
        
        return correlation_matrix, numeric_df
    except FileNotFoundError:
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 로드 및 전처리 중 오류 발생: {e}")
        return pd.DataFrame(), pd.DataFrame()

# 상관관계 분석 함수
def analyze_correlation(corr_matrix):
    """상관관계 행렬에서 가장 높은 양의/음의 상관관계를 찾습니다."""
    
    # 자기 자신과의 상관관계(1)를 제외하고 상삼각 행렬만 사용
    corr_unstack = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()

    # 1. 가장 높은 양의 상관관계
    max_corr = corr_unstack.idxmax()
    max_corr_value = corr_unstack.max()

    # 2. 가장 높은 음의 상관관계 (절댓값이 아닌, 음수 값 중 가장 작은 값)
    min_corr = corr_unstack.idxmin()
    min_corr_value = corr_unstack.min()

    return max_corr, max_corr_value, min_corr, min_corr_value

# 메인 실행 로직
if __name__ == "__main__":
    # 데이터 로드 및 상관관계 행렬 계산
    corr_matrix, numeric_df = load_and_preprocess_data(FILE_NAME)

    if not corr_matrix.empty:
        # 상관관계 분석 수행
        max_corr, max_corr_value, min_corr, min_corr_value = analyze_correlation(corr_matrix)

        st.markdown("---")

        # 💡 가장 높은 양의 상관관계 버튼
        if st.button('⬆️ 가장 높은 양의 상관관계 보기'):
            st.success('**최고 양의 상관관계**')
            st.write(f"두 속성: **{max_corr[0]}**와(과) **{max_corr[1]}**")
            st.write(f"상관계수: **{max_corr_value:.4f}**")
            st.info("값이 1에 가까울수록 한 변수가 증가할 때 다른 변수도 증가하는 경향이 강합니다.")

        st.markdown("---")

        # ⬇️ 가장 높은 음의 상관관계 버튼
        if st.button('⬇️ 가장 높은 음의 상관관계 보기'):
            st.error('**최고 음의 상관관계**')
            st.write(f"두 속성: **{min_corr[0]}**와(과) **{min_corr[1]}**")
            st.write(f"상관계수: **{min_corr_value:.4f}**")
            st.info("값이 -1에 가까울수록 한 변수가 증가할 때 다른 변수는 감소하는 경향이 강합니다.")
            
        st.markdown("---")
        
        # 선택 사항: 전체 상관관계 행렬 표시
        with st.expander("📊 전체 상관관계 행렬 (선택 사항)"):
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format(precision=3))
            
        # 선택 사항: 데이터 샘플 표시
        with st.expander("📋 전처리된 데이터 샘플"):
            st.dataframe(numeric_df.head())
