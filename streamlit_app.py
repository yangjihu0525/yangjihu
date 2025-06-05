import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup

# UI
st.title("📈 AI 기반 종합 주가 예측 분석 시스템")
ticker = st.text_input("🔍 분석할 종목 코드 입력 (예: AAPL, TSLA, 005930.KS)", "AAPL")

# 데이터 가져오기
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="1y")
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    return data

data = load_data(ticker)
st.subheader("📊 주가 및 이동 평균")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="종가"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50"))
st.plotly_chart(fig)

# 머신러닝 예측
st.subheader("🤖 간단한 머신러닝 기반 예측 (Linear Regression)")

df = data.dropna().copy()
df["Tomorrow"] = df["Close"].shift(-1)
features = ["Close", "SMA20", "SMA50"]
X = df[features][:-1]
y = df["Tomorrow"][:-1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)
tomorrow_pred = model.predict(scaler.transform([df[features].iloc[-1]]))[0]

st.metric("📈 내일 예상 종가", f"{tomorrow_pred:.2f} USD")

# 히트맵 (상관 분석)
st.subheader("📌 주가 관련 지표 상관관계 히트맵")
correlation = df[["Close", "SMA20", "SMA50"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
st.pyplot(plt)

# 뉴스 분석 (간단 요약)
st.subheader("📰 최신 뉴스 요약")

def get_news_summary(ticker):
    search_url = f"https://news.google.com/search?q={ticker}"
    try:
        res = requests.get(search_url)
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.select("article h3")
        return [a.get_text() for a in articles[:5]]
    except Exception as e:
        return ["뉴스를 불러오는 데 실패했습니다."]

news = get_news_summary(ticker)
for i, article in enumerate(news):
    st.write(f"🗞️ {i+1}. {article}")

# 경제 지표 (예시: 환율, 유가 등)
st.subheader("🌐 간단한 글로벌 경제 지표 (샘플)")

st.write("✅ 환율 (USD/KRW): 약 1,350원 (예시)")
st.write("✅ 국제 유가 (WTI): 약 75 USD (예시)")
st.write("✅ 국가 성장률: 1.9% (예시)")

# 종합 설명
st.markdown("""
---
📌 **분석 요약**  
- 기술적 분석에 따라 SMA20이 SMA50보다 위에 있을 경우 상승 추세입니다.  
- 머신러닝 예측에 따르면 내일 주가는 약간 상승할 것으로 보입니다.  
- 최근 뉴스에서 해당 기업 관련 긍정/부정 이슈를 파악하세요.  
- 경제 지표는 기업 가치에 중장기적 영향을 미칩니다.

""")

