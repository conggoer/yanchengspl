#coding = utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
path = "C:\\E\\Project\\fusai_need_data\\"
train = pd.read_csv(path + "merge_train.csv")
testA = pd.read_csv(path + "testB1.csv")
workdaylist=list(testA['workday'])
monthlist=list(testA['month'])
dowlist = list(testA['day_of_week'])
tiaoxiulist=list(testA['tiaoxiu'])
enc = OneHotEncoder()
enc.fit(train['day_of_week'].values.reshape(-1,1))
train_enc=enc.transform(train['day_of_week'].values.reshape(-1,1))
enc.fit(testA['day_of_week'].values.reshape(-1,1))
testA_enc=enc.transform(testA['day_of_week'].values.reshape(-1,1))
train_week=train_enc.toarray()
testA_week=testA_enc.toarray()
train_df=pd.DataFrame(train_week,columns=['w1','w2','w3','w4','w5','w6','w7'])
testA_df=pd.DataFrame(testA_week,columns=['w1','w2','w3','w4','w5','w6','w7'])
#====================================================================================
enc2 = OneHotEncoder()
enc2.fit(train['brand'].values.reshape(-1,1))
train_enc2=enc2.transform(train['brand'].values.reshape(-1,1))
enc2.fit(testA['brand'].values.reshape(-1,1))
testA_enc2=enc2.transform(testA['brand'].values.reshape(-1,1))
train_brand=train_enc2.toarray()
testA_brand=testA_enc2.toarray()
train_df2=pd.DataFrame(train_brand,columns=['b1','b2','b3','b4','b5','b6','b7','b8','b9','b10'])
testA_df2=pd.DataFrame(testA_brand,columns=['b1','b2','b3','b4','b5','b6','b7','b8','b9','b10'])
# train=pd.concat([train,train_df2],axis=1)
# testB=pd.concat([testA,testA_df2],axis=1)
# train=train.drop(['day_of_week'],axis=1)
# testB=testB.drop(['day_of_week'],axis=1)
#将month进行onehot编码
enc1 = OneHotEncoder()
enc1.fit(train['month'].values.reshape(-1,1))
train_enc1=enc1.transform(train['month'].values.reshape(-1,1))
testA_enc1=enc1.transform(testA['month'].values.reshape(-1,1))
train_month=train_enc1.toarray()
testA_month=testA_enc1.toarray()
train_df1=pd.DataFrame(train_month,columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'])
testA_df1=pd.DataFrame(testA_month,columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'])
train=pd.concat([train,train_df,train_df1,train_df2],axis=1)
testA=pd.concat([testA,testA_df,testA_df1,testA_df2],axis=1)
date_list = list(testA['date'])
date_list = [int(i) for i in date_list]
brand_list = list(testA['brand'])
brand_list = [int(i) for i in brand_list]
train=train.drop(['month','day_of_week','brand'],axis=1)
testA=testA.drop(['month','day_of_week','brand'],axis=1)
# ansA = pd.read_table(path+'fusai_answer_a_20180307.txt',header=None,engine='python')
# sampleA = pd.read_table(path+"fusai_sample_A_20180227.txt",header=None,engine='python')

fill_number = train[train['workday']==0].cnt.mean()
boundary = int(9*train['date'].max()/10)       #训练集和测试集分界值
boundary = 1100
train_xy = train[(train['date']<=boundary)]
train_val = train[train['date']>boundary]
#对date取对数,对year相对2012处理
# train_xy['date']=np.log(train_xy['date'])
# train_val['date']=np.log(train_val['date'])
y = train_xy.cnt
X = train_xy.drop(['cnt'],axis=1)
val_y = train_val.cnt
val_X = train_val.drop(['cnt'],axis=1)
#testA['date']=np.log(testA['date'])
#0.1x+1000
# X['date']=0.1*X['date']+100
# val_X['date'] = 0.1*val_X['date']+100
#得到评价指标
def MSE_xg(y_hat,y):
    #y DMatrix对象
    y = y.get_label()
    #y_hat = fuzhi_handle(y_hat)
    #y.get_label二维数组
    MSE1=np.mean((y-y_hat)**2)
    return "MSE1",MSE1
#对结果负值处理
def fuzhi_handle(cnt_list):
    for i in range(len(cnt_list)):
        if cnt_list[i]<0:
            cnt_list[i]=fill_number
    return cnt_list
#对预测结果信息修正

#xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X,val_y)
xgb_train = xgb.DMatrix(X,y)
xgb_test = xgb.DMatrix(testA)
#xgb参数
num_boost_round = 200000
watchlist=[(xgb_train,'train'),(xgb_val,'val')]
params = {
    "objective":"reg:linear",
    "reg_alpha":0.01,
    "silent":1,
    "eta":0.01,
    "booster":'gbtree',
    "max_depth":7,
    "subsample":0.7,
    "colsample_bytree":0.9,
    "gamma":0.1,
    "lamda":3,
    "eval_metric":'rmse'
}

#训练模型
model = xgb.train(params,xgb_train,num_boost_round,watchlist,early_stopping_rounds=30000,feval=MSE_xg,verbose_eval=True)
model.save_model('spxgb.model')
xgb.plot_importance(model)

plt.show()
xgb_predictA=model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
xgb_predictval=model.predict(xgb_val,ntree_limit=model.best_ntree_limit)
xgb_predictA=list(xgb_predictA)
xgb_predictval=list(xgb_predictval)
# error=[]
# for i in range(len(xgb_predictval)):
#     error[i] = xgb_predictval[i]-list(train_val['cnt'])[i]
# xgb_predictA = fuzhi_handle(xgb_predictA)
# print(ansA_list)
# print(len(ansA_list))
# print(len(xgb_predictA))
# print(mean_squared_error(xgb_predictA,ansA_list))
# plt.plot(range(len(ansA_list)),ansA_list,range(len(ansA_list)),xgb_predictA)
# plt.legend(['actual','predict'],loc='best')
# plt.show()
plt.plot(range(len(xgb_predictval)),train_val,range(len(xgb_predictval)),xgb_predictval)
plt.show()
xgb_predictA = pd.DataFrame({'date':date_list,'rand':brand_list,'scnt':xgb_predictA,'workday':workdaylist,'month':monthlist,'day_of_week':dowlist,'tiaoxiu':tiaoxiulist})
#对结果做修正
xgb_predictA.loc[(xgb_predictA['rand']==5)|(xgb_predictA['rand']==8),'scnt']=xgb_predictA.loc[(xgb_predictA['rand']==5)|(xgb_predictA['rand']==8),'scnt']*1.2
xgb_predictA.loc[xgb_predictA['rand']==10,'scnt']=xgb_predictA.loc[xgb_predictA['rand']==10,'scnt']*0.9
xgb_predictA.loc[(xgb_predictA['rand']==9)&((xgb_predictA['month']==10)|(xgb_predictA['month']==12)|(xgb_predictA['month']==1)|(xgb_predictA['month']==2)|(xgb_predictA['month']==5)|(xgb_predictA['month']==6)),'scnt']*=1.1
xgb_predictA.loc[(xgb_predictA['rand']==9)&((xgb_predictA['day_of_week']==6)|(xgb_predictA['day_of_week']==7))&(xgb_predictA['tiaoxiu']==0),'scnt']*=0.6
xgb_predictA.loc[(xgb_predictA['workday']==0)&(xgb_predictA['scnt']>60),'scnt']=60
xgb_predictA.loc[(xgb_predictA['scnt']<0),'cnt']=30
xgb_predictA=xgb_predictA.drop(['workday','month','tiaoxiu'],axis=1)
xgb_predictA.to_csv('predictBcnt0309_1.txt',header=False,index=False,sep='\t')
