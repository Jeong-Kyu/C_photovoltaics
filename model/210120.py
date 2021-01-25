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
df_train=df_train.to_numpy()
print(df_train.shape) #(52464,9)
x = df_train[:,:-2]
y = df_train[:,-2:]

x=x.reshape(1093,48,7)
y=y.reshape(1093,48,2)

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
X_test=X_test.reshape(81,48,7)
# print(X_test) # (3888, 7)



from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y,train_size=0.8, random_state=0, shuffle=False)

print(x_train.shape)
print(x_val.shape)
# print(x1_train.shape)
# print(x1_val.shape)
print(y_train.shape)
# print(y1_train.shape)
print(y_val.shape)
# print(y1_val.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D, Dropout, Flatten
from tensorflow.keras.backend import mean, maximum

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def quantile_loss(i, q, y, pred):
  err = (y[i]-pred[i])
  return mean(maximum(q*err, (q-1)*err), axis=-1)

y0=[]
y9=[]
for q in q_lst:
    model = Sequential()
    model.add(Conv1D(filters = 100, kernel_size=1, input_shape=(48,7)))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dropout(0.4))
    model.add(Dense(200))
    model.add(Dense(2))
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)
    
    model.compile(loss=lambda y,pred: quantile_loss(0, q,y,pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(0, q, y, pred)])
    hist1 = model.fit(x_train, y_train, batch_size=96, epochs=1, validation_split=0.2, callbacks=[es, reduce_lr])
    model.evaluate(x_val, y_val, batch_size=96)
    y1_pred = model.predict(X_test)
    y0.append(y1_pred)

    model.compile(loss=lambda y,pred: quantile_loss(1, q,y,pred), optimizer='adam', metrics=[lambda y, pred: quantile_loss(1, q, y, pred)])
    hist1 = model.fit(x_train, y_train, batch_size=96, epochs=1, validation_split=0.2, callbacks=[es, reduce_lr])
    model.evaluate(x_val, y_val, batch_size=96)
    y1_pred = model.predict(X_test)
    y9.append(y1_pred)

y0=np.array(y0)
Y0 = y0.transpose()
Y0 = Y0.reshape(7776,9)
# Y0 = pd.DataFrame(y0)
print(Y0.shape)

y9 =np.array(y9)
Y9 = y9.transpose()
Y9 = Y9.reshape(7776,9)
# Y9 = pd.DataFrame(y9)
print(Y9.shape)

Y5=[]
for w in range(81):
    Y5.append(Y0[(w*48):((w+1)*48),:])
    Y5.append(Y9[(w*48):((w+1)*48),:])

Y5 = np.asarray(Y5)

print(Y5)
print(Y5.shape)
Y5 = Y5.reshape(15552,9)
Y5 = pd.DataFrame(Y5)

index_c = []

for i in range(81):
    for a in range(2):
        for b in range(24):
            for c in range(2):
                index = str(i)+".csv_Day"+str(a+7)+"_"+str(b)+"h"+"%02d"%(30*(c))+"m"
                index_c.append(index)
Y5.columns = ['q_0.1','q_0.2','q_0.3','q_0.4','q_0.5','q_0.6','q_0.7','q_0.8','q_0.9']
Y5.index = index_c                
print(Y5)
Y5.to_csv('./csv/test1.csv', index=True)