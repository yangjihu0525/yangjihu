import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression

st.title("📈 AI 기반 종합 주가 분석 시스템")

# --- 종목 입력 ---
ticker = st.text_input("🔎 종목 코드 입력 (예: AAPL, TSLA, 005930.KS)", "AAPL")

if ticker:
    data = yf.download(ticker, period="6mo")
    st.subheader("💹 주가 차트 (6개월)")
    st.line_chart(data["Close"])

    # 기술적 지표
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA50"))
    st.plotly_chart(fig)

    # 단기 주가 예측 (선형 회귀)
    st.subheader("🔮 단기 주가 예측")
    df = data[["Close"]].copy().dropna()
    df["Prediction"] = df["Close"].shift(-5)
    X = np.array(df[["Close"]][:-5])
    y = np.array(df["Prediction"][:-5])
    model = LinearRegression()
    model.fit(X, y)
    future = model.predict(np.array(df[["Close"]][-5:]))

    st.write("**다음 주 예상 종가 (단순 회귀 기반):**")
    for i, val in enumerate(future):
        st.write(f"📅 {i+1}일 후 예상: {round(val, 2)}")

    st.markdown("---")

    # --- 경제 지표 ---
    st.subheader("🌍 주요 경제 지표")

    # 환율 (예: USD/KRW)
    try:
        fx_url = "https://api.exchangerate.host/latest?base=USD&symbols=KRW"
        fx_res = requests.get(fx_url).json()
        usd_krw = fx_res["rates"]["KRW"]
        st.metric(label="환율 (USD/KRW)", value=round(usd_krw, 2))
    except:
        st.warning("환율 정보를 가져올 수 없습니다.")

    # 미국 10년 국채 금리 (대략 값 가져오기)
    try:
        treasury_url = "https://datahub.io/core/interest-rates/r/10-year-us-treasury-rate.csv"
        treasury_df = pd.read_csv(treasury_url)
        latest_rate = treasury_df.iloc[-1]["Value"]
        st.metric(label="미국 10년 국채 금리 (%)", value=latest_rate)
    except:
        st.warning("금리 정보를 가져올 수 없습니다.")

    # 소비자 물가지수(CPI), 실업률 같은 데이터는 공공 API 등 별도 연결 필요

    # --- 정치 이슈 뉴스 ---
    st.subheader("📰 정치·경제 이슈 뉴스")
    NEWS_API_KEY = "여기에_뉴스API키_넣기"
    if NEWS_API_KEY != "여기에_뉴스API키_넣기":
        query = "정치 OR 경제"
        news_url = f"https://newsapi.org/v2/everything?q={query}&language=ko&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        res = requests.get(news_url).json()
        articles = res.get("articles", [])[:5]
        for art in articles:
            st.markdown(f"- [{art['title']}]({art['url']})")
    else:
        st.warning("뉴스API 키를 입력해 주세요.")

