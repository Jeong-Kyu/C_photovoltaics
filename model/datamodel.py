import numpy as np
import pandas as pd

train_data = pd.read_csv('./csv/train/train.csv',thousands = ',', encoding='UTF8', index_col=0, header=0)

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        
        return temp.iloc[:-96]

    elif is_train==False:
        
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
                              
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

# print(X_test.shape) # (3888, 7)


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


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(df_train.iloc[:,:-2], df_train.iloc[:,-2],train_size=0.8, random_state=0)
x1_train, x1_val, y1_train, y1_val = train_test_split(df_train.iloc[:,:-2], df_train.iloc[:,-1],train_size=0.8, random_state=0)

print(x_train.shape) # (41971, 7)
print(y_train.shape) # (41971, )


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(Dense(10, input_shape=(7, ), activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val loss', patience=30)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, verbose=1)
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
model.fit(x_train, y_train, batch_size=96, epochs=1000, validation_split=0.2, callbacks=[es, reduce_lr])
model.evaluate(x_val, y_val, batch_size=96)
y1_pred = model.predict(X_test)
print(y1_pred)
'''
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

    # y_p1 = y1_pred[a]
    # percenti1 = np.percentile(y_p1, [0,10,20,30,40,50,60,70,80,90], interpolation='nearest')
    # y0.append(percenti1)
    # y_p2 = y2_pred[a]
    # percenti2 = np.percentile(y_p2, [0,10,20,30,40,50,60,70,80,90], interpolation='nearest')
    # y0.append(percenti2)

Y0 = pd.DataFrame(y0)

index_c = []

for i in range(81):
    for a in range(2):
        for b in range(24):
            for c in range(2):
                index = str(i)+".csv_Day"+str(a+7)+"_"+str(b)+"h"+"%02d"%(30*(c))+"m"
                index_c.append(index)
Y0.columns = ['q_0.1','q_0.2','q_0.3','q_0.4','q_0.5','q_0.6','q_0.7','q_0.8','q_0.9']
Y0.index = index_c                
print(Y0)
Y0.to_csv('./csv/test1.csv', index=True)
'''


'''

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

'''