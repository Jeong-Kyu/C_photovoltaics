# 데이터 각 특성별 

import numpy as np
import pandas as pd

train_data = pd.read_csv('./csv/train/train.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)
submission = pd.read_csv('./csv/sample_submission.csv')

def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI','DNI', 'RH', 'T']]# 'WS',

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False:
        temp = temp[['Hour', 'TARGET', 'DHI','DNI', 'RH', 'T']]# 'WS',                    
        return temp.iloc[-48:, :]

df_train = preprocess_data(train_data)
df_train=df_train.to_numpy()

df_test = []
for i in range(81):
    file_path = './csv/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)
x_test = pd.concat(df_test)
x_test=x_test.to_numpy()

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))
x,y1,y2 = split_xy(df_train,1)

def split_x(data,timestep):
    x = []
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end]
        x.append(tmp_x)
    return(np.array(x))
x_test = split_x(x_test,1)

from sklearn.model_selection import train_test_split
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x,y1,y2,train_size=0.8, random_state=0, shuffle=False)
x_train = x_train.reshape(41971, 6)
x_val = x_val.reshape(10493, 6)
x_test = x_test.reshape(3888, 6)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(41971,1, 6)
x_val = x_val.reshape(10493, 1,6)
x_test = x_test.reshape(3888, 1,6)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D, Dropout, Flatten
from tensorflow.keras.backend import mean, maximum

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis=-1)

y0=[]
y1=[]
for q in q_lst:
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape = (1,6)))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])
    model.fit(x_train, y1_train, batch_size=96, epochs=100, validation_data=(x_val, y1_val), callbacks=[es, reduce_lr])
    model.evaluate(x_val, y1_val, batch_size=96)
    y1_pred = model.predict(x_test)
    y1_pred = pd.DataFrame(y1_pred)
    y0.append(y1_pred)
df_temp1 = pd.concat(y0, axis = 1)
df_temp1[df_temp1<0] = 0
num_temp1 = df_temp1.to_numpy()
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = num_temp1

for q in q_lst:
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape = (1,6)))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
    model.compile(loss=lambda y,pred: quantile_loss(q,y,pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(q, y, pred)])    
    model.fit(x_train, y2_train, batch_size=96, epochs=100, validation_data=(x_val, y2_val),callbacks=[es, reduce_lr])
    model.evaluate(x_val, y2_val, batch_size=96)
    y2_pred = model.predict(x_test)
    y2_pred = pd.DataFrame(y2_pred)
    y1.append(y2_pred)
df_temp2 = pd.concat(y1, axis = 1)
df_temp2[df_temp2<0] = 0
num_temp2 = df_temp2.to_numpy()
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = num_temp2
        
submission.to_csv('./csv/sample_submission2101200.csv', index = False)

# loss: 0.8082 - <lambda>: 0.8082