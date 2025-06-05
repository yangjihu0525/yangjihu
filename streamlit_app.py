import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

st.title("📊 AI 기반 종합 주가 분석 & 머신러닝 예측 앱")

ticker = st.text_input("종목 코드 입력 (예: AAPL, TSLA)", "AAPL")

if ticker:
    # Yahoo Finance에서 데이터 불러오기 (최근 6개월)
    data = yf.download(ticker, period="6mo")

    if not data.empty:
        st.subheader(f"{ticker} 최근 6개월 종가 차트")
        st.line_chart(data["Close"])

        # 머신러닝 예측 부분
        st.subheader("📈 머신러닝 주가 예측 (Linear Regression)")

        data_ml = data.reset_index()
        data_ml['DateOrdinal'] = data_ml['Date'].map(lambda x: x.toordinal())

        X = data_ml[['DateOrdinal']]
        y = data_ml['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        fig, ax = plt.subplots()
        ax.plot(data_ml['Date'][-len(y_test):], y_test, label='Actual')
        ax.plot(data_ml['Date'][-len(y_test):], y_pred, label='Predicted')
        ax.set_title(f'Linear Regression 예측 (RMSE: {rmse:.2f})')
        ax.legend()

        st.pyplot(fig)

    else:
        st.error("해당 종목의 데이터를 가져올 수 없습니다. 종목 코드를 확인해주세요.")
else:
    st.info("위에 종목 코드를 입력해주세요.")
