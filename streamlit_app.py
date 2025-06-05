import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 AI 기반 종합 주가 분석 시스템")

# 종목 입력
ticker = st.text_input("종목 코드 입력 (예: AAPL, TSLA, MSFT)", "AAPL")
data = yf.download(ticker, period="6mo")

# 주가 차트
st.subheader(f"📊 {ticker} 최근 6개월 주가")
st.line_chart(data["Close"])

# 기술적 지표
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="종가"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50"))
st.plotly_chart(fig)

# 산업군 비교 (FAANG 기준)
faang = ["AAPL", "AMZN", "GOOGL", "META", "NFLX"]
faang_data = yf.download(faang, period="6mo")["Close"]

st.subheader("🧾 FAANG 종목 비교")
st.line_chart(faang_data)

# 뉴스 요약 (뉴스 API 필요, 아래는 뉴스API.org 사용 예시)
st.subheader("📰 최근 뉴스 헤드라인")
NEWS_API_KEY = "your_newsapi_key_here"  # NewsAPI 키 입력
if NEWS_API_KEY != "your_newsapi_key_here":
    news_url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    res = requests.get(news_url).json()
    articles = res["articles"][:5]
    for article in articles:
        st.markdown(f"- [{article['title']}]({article['url']})")
else:
    st.warning("🔑 News API 키를 입력해 주세요 (NEWS_API_KEY).")

# 간단한 선형 회귀 예측
st.subheader("🤖 AI 예측 (선형 회귀)")
data = data.dropna()
X = np.arange(len(data)).reshape(-1, 1)
y = data["Close"].values
model = LinearRegression().fit(X, y)
future_X = np.arange(len(data) + 5).reshape(-1, 1)
future_y = model.predict(future_X)

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=data.index, y=data["Close"], name="실제"))
future_dates = pd.date_range(start=data.index[-1], periods=6, freq="D")
fig_pred.add_trace(go.Scatter(x=list(data.index) + list(future_dates), y=future_y, name="예측"))
st.plotly_chart(fig_pred)
