import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📈 AI 기반 주가 분석 앱")

# 사용자 입력
ticker = st.text_input("🔍 종목 코드를 입력하세요 (예: AAPL, TSLA, MSFT)", "AAPL")

# 데이터 불러오기
data = yf.download(ticker, period="6mo")

if data.empty:
    st.error("❌ 유효한 종목 코드가 아닙니다.")
    st.stop()

# 주가 차트
st.subheader("📉 최근 6개월 주가")
st.line_chart(data["Close"])

# 기술적 지표: 이동 평균선
st.subheader("📊 기술적 지표 (이동 평균선)")
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50"))
fig.update_layout(title=f"{ticker} 주가 및 이동 평균선", xaxis_title="날짜", yaxis_title="가격")
st.plotly_chart(fig)

# 산업군 비교 기능
industry_map = {
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "TSLA": ["GM", "F", "NIO"],
    "NVDA": ["AMD", "INTC", "QCOM"]
}

st.subheader("📂 산업군 비교")

if ticker in industry_map:
    peer_list = industry_map[ticker]
    st.markdown(f"💡 `{ticker}`는 `{', '.join(peer_list)}` 와 같은 산업군에 속합니다.")
    
    peer_data = yf.download(peer_list, period="6mo")["Close"]
    peer_data[ticker] = data["Close"]
    peer_data = peer_data.dropna()
    
    st.line_chart(peer_data)
    
    st.subheader("📌 산업군 종목 간 상관관계")
    corr = peer_data.corr()
    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)
else:
    st.warning("⛔ 이 종목의 산업군 데이터가 등록되어 있지 않습니다.")
