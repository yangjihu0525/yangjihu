import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

st.subheader("📈 머신러닝 주가 예측")

# 데이터 전처리: 날짜를 숫자형으로 변환 (예: timestamp)
data_ml = data.reset_index()
data_ml['DateOrdinal'] = data_ml['Date'].map(lambda x: x.toordinal())

# Feature, Target 정의
X = data_ml[['DateOrdinal']]
y = data_ml['Close']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 오차 계산
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 시각화
import matplotlib.pyplot as plt

fig2, ax = plt.subplots()
ax.plot(data_ml['Date'][-len(y_test):], y_test, label='Actual')
ax.plot(data_ml['Date'][-len(y_test):], y_pred, label='Predicted')
ax.set_title(f'Linear Regression 예측 (RMSE: {rmse:.2f})')
ax.legend()

st.pyplot(fig2)
