# 데이터 각 특성별 

import numpy as np
import pandas as pd

train_data = pd.read_csv('./csv/train/train.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)
submission = pd.read_csv('./csv/sample_submission.csv')

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS','RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS','RH', 'T']]
                              
        return temp.iloc[-48:, :]
df_train = preprocess_data(train_data)

# df_train.reshape(52464,9,1)

# df_train = df_train.astype('float32')

# df_train=df_train.to_numpy()

df_test = []

for i in range(81):
    file_path = './csv/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)
# print(df_test)
# print('--------------------')
X_test = pd.concat(df_test)
X_test=X_test.to_numpy()
x_test = X_test
# print(X_test) # (3888, 7)


# print(df_train)
# def split_xy4(dataset, x_low, x_col, y_low, y_col):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + x_low
#         y_end_number = x_end_number + y_low -1
#         if y_end_number > len(dataset):
#             break
#         tmp_x = dataset[i:x_end_number,:x_col]
#         tmp_y = dataset[x_end_number:y_end_number+1,x_col:]
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)

# x, y = split_xy4(df_train, 7*48,7,2*48,2)
# print(type(x))
# print(type(y))
def same_train(train) :
    temp = train.copy()
    x = list()
    final_x = list()
    for i in range(48) :
        same_time = pd.DataFrame()
        for j in range(int(len(temp)/48)) :
            tmp = temp.iloc[i + 48*j, : ]
            tmp = tmp.to_numpy()
            tmp = tmp.reshape(1, tmp.shape[0])
            tmp = pd.DataFrame(tmp)
            # print(tmp)
            same_time = pd.concat([same_time, tmp])
        x = same_time.to_numpy()
        final_x.append(x)
    return np.array(final_x)

same_time = same_train(df_train)
# print(same_time.shape) # (48, 1093, 9)

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:,:-2]
        tmp_y1 = data[x_end-1:x_end,:,-2]
        tmp_y2 = data[x_end-1:x_end,:,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

x,y1,y2 = split_xy(same_time,1)

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


# print(x.shape)#(48, 1, 1093, 7)
# print(y1.shape)#(48, 1, 1093)
# print(y2.shape)#(48, 1, 1093)

from sklearn.model_selection import train_test_split
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x,y1,y2,train_size=0.8, random_state=0, shuffle=False)
# print(x_test.shape) # (3888, 1,7)
# print(x_train.shape) # (38, 1, 1093,7)
# print(x_val.shape) # (10, 1, 1093, 7)
# print(y1_train.shape) # (38, 1, 1093)
# print(y1_val.shape) # (10, 1, 1093)
y1_train = y1_train.reshape(41534,1)
y2_train = y1_train.reshape(41534,1)
y1_val = y1_val.reshape(10930,1)
y2_val = y2_val.reshape(10930,1)

x_train = x_train.reshape(41534, 7)
x_val = x_val.reshape(10930, 7)
x_test = x_test.reshape(3888, 7)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# print(x_train.shape) # (41971, 1,7)
x_train = x_train.reshape(41534,1, 7)
x_val = x_val.reshape(10930, 1,7)
x_test = x_test.reshape(3888, 1,7)
# print(y_train.shape) # (41971, )
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
# X_test = scaler.transform(X_test)
# x1_train = scaler.transform(x_train)
# x1_val = scaler.transform(x_val)

# x_train = x_train.reshape(41971,7,1)
# x_val = x_val.reshape(10493,7,1)
# X_test = X_test.reshape(3888,7,1)
# x1_train = x1_train.reshape(41971,7,1)
# x1_val = x1_val.reshape(10493,7,1)
# print(y_train)
# print(y1_train)


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
    # model.add(LSTM(256, activation='relu', input_shape = (1,7)))
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (1,7)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.7, verbose=1)
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
    # model.add(LSTM(256, activation='relu', input_shape = (1,7)))
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (1,7)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.7, verbose=1)
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
        
submission.to_csv('./csv/sample_submission210125.csv', index = False)

'''
y0=np.array(y0)    
print(y0.shape)
y0 = y0.transpose()
y0 = y0.reshape(3888, 18)
Y0 = pd.DataFrame(y0)
Y0.to_csv('./csv/test1.csv', index=True)

y_percentiles=[]
for a in range(7776):
    percenti1 = np.percentile(Y0, [10,20,30,40,50,60,70,80,90], interpolation='nearest')
    y_percentiles.append(percenti1)
    print(y_percentiles)
    print("7776 / ",a)
y_percentiles = pd.DataFrame(y_percentiles)

index_c = []

for i in range(81):
    for a in range(2):
        for b in range(24):
            for c in range(2):
                index = str(i)+".csv_Day"+str(a+7)+"_"+str(b)+"h"+"%02d"%(30*(c))+"m"
                index_c.append(index)

Y0.columns = ['q_0.1','q_0.2','q_0.3','q_0.4','q_0.5','q_0.6','q_0.7','q_0.8','q_0.9']
y_percentiles.index = index_c                
print(Y0)
Y0.to_csv('./csv/test1.csv', index=True)
'''
'''
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False 
matplotlib.rcParams['font.family'] = "Malgun Gothic"
plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)  # 2행 1열 중 첫번째
plt.plot(hist1.history['loss'], marker= '.', c='red', label='loss')
plt.plot(hist1.history['val_loss'], marker= '.', c='blue', label='val_loss')
plt.grid()
plt.subplot(2,1,2)  # 2행 1열 중 첫번째
plt.plot(hist2.history['loss'], marker= '.', c='red', label='loss')
plt.plot(hist2.history['val_loss'], marker= '.', c='blue', label='val_loss')
plt.grid()
plt.show()


y_pred = np.c_[X_test, y1_pred, y2_pred]
# print(y_pred)

index_c = []

for i in range(81):
    for a in range(2):
        for b in range(24):
            for c in range(2):
                index = str(i)+".csv_Day"+str(a+7)+"_"+str(b)+"h"+"%02d"(30*(c))+"m"
                index_c.append(index)

print(y0)

y_pred = pd.DataFrame(y_pred)
print(y_pred.shape)
y_pred.columns = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
y_pred.index = y0
y_pred.to_csv('./csv/test.csv', index=False)
print(y_pred.shape)




y0=[]
for a in range(3888):
    y_pi = model.predict(X_test[a:a+1,:])
    percenti1 = np.percentile(y_pi, [10,20,30,40,50,60,70,80,90], interpolation='nearest')
    y0.append(percenti1)
    print("3888 / ",a)

model.compile(loss='mse', optimizer='adam', metrics='accuracy')
model.fit(x1_train, y1_train, batch_size=48, epochs=1)
model.evaluate(x1_val, y1_val, batch_size=48)
y2_pred = model.predict(X_test)

for a in range(3888):
    y_pi = model.predict(X_test[a:a+1,:])
    percenti1 = np.percentile(y_pi, [10,20,30,40,50,60,70,80,90], interpolation='nearest')
    y0.append(percenti1)
    print("3888 / ",a)


Y0 = pd.DataFrame(y0)
'''

#  LSTM - StandardScaler
#  loss: 0.8108 - <lambda>: 0.8108
#  LSTM - MinMaxScaler
#  loss: 0.8211 - <lambda>: 0.8211
#  Conv1d - StandardScaler
#  loss: 0.8201 - <lambda>: 0.8201
#  Conv1d - StandarScaler - Dropout - epochs=10
#  loss: 1.3
#  Conv1d - StandarScaler - Dropout - epochs=10 - RH제외(상관계수 최하)
#  loss: 1.2044 - <lambda>: 1.2044
#  Conv1d - StandarScaler - Dropout - epochs=100 - RH제외(상관계수 최하)
#  loss: 0.8354 - <lambda>: 0.8354
#  Conv1d - StandarScaler - Dropout - epochs=100
#  loss: 0.8157 - <lambda>: 0.8157

#  1일 기준으로 잡았을때 lose가 너무 낮게나와서,,
#  loss: 0.0141 - <lambda>: 0.0141 (ep 10)
#  loss: 0.0333 - <lambda>: 0.0333 (ep 100) 
#  loss: 0.0242 - <lambda>: 0.0242 (ep 100. Rl=0.7)