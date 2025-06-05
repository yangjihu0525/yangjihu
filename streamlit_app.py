import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.title("📈 다중 종목 주가 비교 및 산업군 분석")

# 여러 종목을 콤마(,)로 구분해서 입력
tickers_input = st.text_input("분석할 종목 코드들을 입력하세요 (예: AAPL,TSLA,MSFT)", "AAPL,TSLA,MSFT")

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) > 0:
    data = yf.download(tickers, period="6mo")["Close"]
    st.subheader("최근 6개월 종가 비교")
    st.line_chart(data)

    st.subheader("종목별 수익률 계산 (6개월간)")
    returns = data.pct_change().dropna()
    cumulative_returns = (1 + returns).prod() - 1
    st.dataframe(cumulative_returns.to_frame("6개월 누적 수익률"))

    st.subheader("종목 간 상관관계 분석")
    corr = returns.corr()
    st.dataframe(corr)

    # 상관관계 히트맵 그리기
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("최소 한 종목 코드를 입력하세요.")
