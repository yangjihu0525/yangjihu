import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI 주가 분석기", layout="wide")
st.title("📊 AI 기반 주가 분석 앱 (1단계: 산업군 비교 분석 포함)")

# 사용자 입력
ticker = st.text_input("🔍 종목 코드를 입력하세요 (예: AAPL, TSLA, MSFT)", "AAPL")

# 데이터 다운로드
data = yf.download(ticker, period="6mo")
stock = yf.Ticker(ticker)
info = stock.info

# 기본 주가 차트
st.subheader(f"📈 {ticker} 최근 6개월 주가")
st.line_chart(data["Close"])

# 기술적 지표
st.subheader("📏 기술적 지표 (단순 이동 평균)")
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50"))
st.plotly_chart(fig)

# 산업군 비교 분석
st.subheader("🏭 산업군 비교 분석")

# 간단한 산업군 매핑 예시
industry_map = {
    "AAPL": ("Technology", ["MSFT", "GOOGL", "NVDA"]),
    "TSLA": ("Automotive", ["GM", "F", "NIO"]),
    "JPM": ("Finance", ["BAC", "C", "WFC"]),
    "PFE": ("Healthcare", ["JNJ", "MRK", "ABBV"])
}

sector, peers = industry_map.get(ticker.upper(), ("Unknown", []))

if sector != "Unknown":
    st.markdown(f"**📂 산업군**: {sector}")
    st.markdown(f"**📊 비교 종목**: {', '.join(peers)}")

    def get_basic_info(ticker):
        try:
            s = yf.Ticker(ticker).info
            return {
                "Ticker": ticker,
                "Price": s.get("currentPrice", None),
                "PER": s.get("trailingPE", None),
                "MarketCap": s.get("marketCap", None)
            }
        except:
            return {"Ticker": ticker, "Price": None, "PER": None, "MarketCap": None}

    compare_data = [get_basic_info(ticker)] + [get_basic_info(p) for p in peers]
    df = pd.DataFrame(compare_data)

    st.dataframe(df.set_index("Ticker"))

    # 시가총액 비교 차트
    st.subheader("💰 시가총액 비교 (단위: 조)")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=df["Ticker"], y=[v/1e12 if v else 0 for v in df["MarketCap"]], name="Market Cap"))
    fig2.update_layout(yaxis_title="조 단위 (T)", title="Market Cap 비교")
    st.plotly_chart(fig2)

    # PER 비교 차트
    st.subheader("📐 PER 비교")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=df["Ticker"], y=df["PER"], name="PER"))
    fig3.update_layout(title="PER (주가수익비율) 비교")
    st.plotly_chart(fig3)
else:
    st.warning("⛔ 이 종목의 산업군 데이터가 등록되어 있지 않습니다.")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("최소 한 종목 코드를 입력하세요.")
