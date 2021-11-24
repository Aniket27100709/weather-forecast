import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet

data=pd.read_csv('weatherAUS.csv')

data1=data[data['Location']=='Albury']
data1['Date']=pd.to_datetime(data1['Date'])

data1['Year']=data1['Date'].apply(lambda x:x.year)
data2=data1[data1['Year']<=2015]

data=data2[['Date','Temp3pm']]
data.dropna(inplace=True)
data.columns=['ds','y']

np=NeuralProphet()
np.fit(data,freq='D',epochs=1000)


st.title('Weather Forecasting Application')

st.write("SELECT FORECAST PERIOD")
periods_input = st.number_input('How many days forecast do you want?',
 min_value = 1, max_value = 10000)
if st.button('Forecast'):
    future=np.make_future_dataframe(data,periods=periods_input)
    forecast=np.predict(future)
    plotting=np.plot(forecast)
    plotting2=np.plot_components(forecast)
    st.write(plotting)
    st.write(plotting2)
st.image('23284.jpg')