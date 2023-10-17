import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from keras.models import load_model
from PIL import Image

#Loading image and chhanging page icon img
img=Image.open('/home/harsh/Documents/Stock.jpg')
st.set_page_config(page_title="StockPredict",page_icon=img)


start = '2013-01-01'
end = '2023-10-12'

st.title('Stock Trend Prediction')

user_input=st.text_input('Enter Stock ticker','AXISBANK.NS')

if st.button('predict'):

    df = yf.download(user_input, start ,end )
   
   
   #describe 
    st.subheader('Data from 2013 - 2023')
    st.write(df.describe())



    #maps
    st.subheader('Closing Price vs Time Chart')
    fig=plt.figure(figsize= (12,6))
    plt.plot(df.Close,color='b')
    plt.legend()
    st.pyplot(fig)


    st.subheader('closing price VS Time Chart with 100 moving Average')
    ma100=df.Close.rolling(100).mean()
    fig= plt.figure(figsize=(12,6))
    plt.plot(ma100,color='r')
    plt.plot(df.Close)
    plt.legend()
    st.pyplot(fig)


    st.subheader('closing price VS Time Chart with 100 moving Average and 200 moving average')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig= plt.figure(figsize=(12,6))
    plt.plot(ma100,color='r')
    plt.plot(ma200,color='g')
    plt.plot(df.Close)
    plt.legend()
    st.pyplot(fig)


    #spltting data into train test 
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0,1))

    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100 ,data_training_array.shape[0]):
      x_train.append(data_training_array[i-100: i])
      y_train.append(data_training_array[i,0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)

    model= load_model('model.h5')


    #testing
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100 , input_data.shape[0]):
      x_test.append(input_data[i-100:i])
      y_test.append(input_data[i,0])
    
    
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler=scaler.scale_
    
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor

    st.subheader('Predicted VS Original')
    fig = plt.figure(figsize= (12,6))
    plt.plot(y_test , 'b', label = 'Original Price')
    plt.plot(y_predicted , 'r', label = 'prdicted Price')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend()
    plt.show()
    st.pyplot(fig)


  









            

