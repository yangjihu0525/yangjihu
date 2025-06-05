import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("📊 AI 기반 주가 분석 앱")

ticker = st.text_input("종목 코드를 입력하세요 (예: AAPL, TSLA)", "AAPL")
data = yf.download(ticker, period="6mo")

st.subheader("최근 6개월 주가")
st.line_chart(data["Close"])

st.subheader("기술적 지표 (단순 이동 평균)")
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50"))

st.plotly_chart(fig)
