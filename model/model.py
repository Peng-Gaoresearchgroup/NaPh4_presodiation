import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error
import pandas as pd
import xgboost as xgb
torch.manual_seed(42)

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        fc_hidden_dim=32
        hidden_chan=1
        width=41
        hight=6

        self.embedding = nn.Embedding(278, width)
        self.debeddding= nn.Conv2d(1, hidden_chan, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv1 = nn.Conv2d(hidden_chan, hidden_chan, kernel_size=(3,1), stride=(1,1), padding=(2,0)) # hight 4 -> 6
        self.conv2 = nn.Conv2d(hidden_chan, hidden_chan, kernel_size=(2,1), stride=(1,1), padding=(0,0)) # hight 3 -> 2
        self.conv6 = nn.Conv2d(hidden_chan, 1, kernel_size=(1,1), stride=(1,1), padding=(0,0)) # channel ? -> 1

        # 定义全连接层
        self.fc1 = nn.Linear(width, fc_hidden_dim)
        self.fc3 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0)
        x=self.debeddding(x)


        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=(2, 1), stride=(2, 1))

        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=(2, 1), stride=(2, 1))
           
        x = F.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc3(x))


        
        x = self.fc2(x)    
        return x

    def _train(self,train_dataset,test_dataloader,train_dataloader_all):
        self.train()
        # train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        # test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        
        # train_dataloader_all = DataLoader(train_dataset, batch_size=4, shuffle=True)
        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        for epoch in range(1000):  # 假设训练1000轮
            train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            for x_train, y_train in train_dataloader:
                optimizer.zero_grad()
                y_pred = self.forward(x_train)
                loss = criterion(y_pred, y_train)
                # arccuracy=accuracy_score(y_pred, y_train)
                loss.backward()
                optimizer.step()
            if epoch==0 or (epoch + 1) % 10 == 0 :
                train_loss,test_loss=self._monitor(test_dataloader=test_dataloader,train_dataloader_all=train_dataloader_all)
                # test_loss=self._test(test_dataloader)
                # train_loss=self._train_loss(train_dataloader_all)
                print(f'{epoch+1},{loss.item():.4f},{train_loss:.9f},{test_loss:.9f}')
                # print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    # def predict()
    def _monitor(self,test_dataloader,train_dataloader_all):
        test_loss=[]
        train_loss=[]
        criterion = nn.L1Loss()
        for x, y in train_dataloader_all:
            y_pred = self.forward(x)
            train_loss.append(criterion(y_pred, y))
        
        for x, y in test_dataloader:
            y_pred = self.forward(x)
            test_loss.append(criterion(y_pred, y))
        
        return sum(train_loss) / len(train_loss),  sum(test_loss) / len(test_loss)

class MyMLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,depth,batch_size,X_train,Y_train,X_test,Y_test):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.X_train=X_train
        self.Y_train=Y_train
        self.hidden_dim=hidden_dim
        self.X_test=X_test
        self.Y_test=Y_test
        self.train_dataset=TensorDataset(torch.tensor(self.X_train.values, dtype=torch.float32),torch.tensor(self.Y_train.values, dtype=torch.float32))
        self.test_dataset=TensorDataset(torch.tensor(self.X_test.values, dtype=torch.float32),torch.tensor(self.Y_test.values, dtype=torch.float32))

        self.depth=depth
        self.batch_size=batch_size

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc23 = nn.Linear(hidden_dim, hidden_dim)
        self.fc24 = nn.Linear(hidden_dim, hidden_dim)
        self.fc25 = nn.Linear(hidden_dim, hidden_dim)
        self.fc26 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        x = self.fc1(x)
        if self.depth >=1:
            x=F.relu(self.fc21(x))
        if self.depth >=2:
            x=F.relu(self.fc22(x))
        if self.depth >=3:
            x=F.relu(self.fc23(x))
        if self.depth >=4:
            x=F.relu(self.fc24(x))
        if self.depth >=5:
            x=F.relu(self.fc25(x))
        if self.depth >=6:
            x=F.relu(self.fc26(x))
        x=self.fc3(x)
        return x
    def _train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(1000):
            for x_train, y_train in train_dataloader:
                optimizer.zero_grad()
                y_pred = self.forward(x_train)
                loss = criterion(y_pred, y_train)
                # arccuracy=accuracy_score(y_pred, y_train)
                loss.backward()
                optimizer.step()
            if epoch==0 or (epoch + 1) % 10 == 0 :
                print(f'{epoch},{loss}')
        # self.eval()
        return loss
    def _monitor(self):
        self.eval()
        test_loss=[]
        train_loss=[]
        criterion = nn.L1Loss()
        for x, y in DataLoader(self.train_dataset, batch_size=len(self.X_train), shuffle=True):
            y_pred = self.forward(x)
            train_loss.append(criterion(y_pred, y))
        
        for x, y in DataLoader(self.test_dataset, batch_size=len(self.X_test), shuffle=True):
            y_pred = self.forward(x)
            test_loss.append(criterion(y_pred, y))
        # print(train_loss,test_loss)
        return sum(train_loss) / len(train_loss),  sum(test_loss) / len(test_loss)
    def _dataset_predict(self):
        self.eval()
        for x, y in DataLoader(self.train_dataset, batch_size=len(self.X_train), shuffle=True):
            y_pred_train = self.forward(x)
        for x, y in DataLoader(self.test_dataset, batch_size=len(self.X_test), shuffle=True):
            y_pred_test = self.forward(x)
        return y_pred_train.detach().numpy().squeeze(),y_pred_test.detach().numpy().squeeze()

class NNOptimizer():
    def __init__(self,input_dim,output_dim,batch_size,X_train,Y_train,X_test,Y_test):
        self.input_dim=input_dim
        self.output_dim=output_dim

        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test
        self.batch_size=batch_size
    def hyperpara_opt(self):
        record=[]
        for hidden_dim in [64,128,256,512,1024]:
            for depth in[1,2,3,4,5,6]:
        # for hidden_dim in [1024]:
        #     for depth in[1]:
                nn=MyMLP(input_dim=self.input_dim,hidden_dim=hidden_dim,output_dim=self.output_dim,depth=depth,
                        batch_size=self.batch_size,
                        X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
                nn._train()
                train_loss,test_loss=nn._monitor()
                record.append({'hidden_dim':hidden_dim,'depth':depth,'mae_train':train_loss.item(),'mae_test':test_loss.item()})
        self.hyperpara_mae=pd.DataFrame(record)
        return pd.DataFrame(record)
    
    def determine_hyperpara(self):
        df=self.hyperpara_mae
        self.best_hidden_dim,self.best_depth=df.loc[df['mae_test'].idxmin(), ['hidden_dim', 'depth']]
        self.best_hidden_dim,self.best_depth=int(self.best_hidden_dim),int(self.best_depth)
        return self.best_hidden_dim,self.best_depth
    
    def get_optmial_model(self):
        hidden_dim,depth=self.determine_hyperpara()
        nn=MyMLP(input_dim=self.input_dim,hidden_dim=hidden_dim,output_dim=self.output_dim,depth=depth,
                        batch_size=self.batch_size,
                        X_train=self.X_train,X_test=self.X_test,Y_train=self.Y_train,Y_test=self.Y_test)
        nn._train()
        self.train_loss,self.test_loss=nn._monitor()
        self.y_train_pre,self.y_test_pre=nn._dataset_predict()
        self.dataset_predict=pd.DataFrame()

        self.mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=self.y_test_pre)
        self.r2_train=r2_score(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.r2_test=r2_score(y_true=self.Y_test,y_pred=self.y_test_pre)
        self.rmse_train=root_mean_squared_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.rmse_test=root_mean_squared_error(y_true=self.Y_test,y_pred=self.y_test_pre)
        if 1:
            max_len=400
            def pad(y,max_len):
                y=np.pad(y, (0, max_len - len(y)), constant_values=np.nan)
                return y
            # print(self.Y_train.shape)
            self.dataset_predict['y_train']=pad(self.Y_train,max_len)
            self.dataset_predict['y_train_pre']=pad(self.y_train_pre,max_len)
            self.dataset_predict['y_test']=pad(self.Y_test,max_len)
            self.dataset_predict['y_test_pre']=pad(self.y_test_pre,max_len)
        else:
            y_all=np.concatenate([self.Y_train, self.Y_test])
            y_all_pre=np.concatenate([self.y_train_pre, self.y_test_pre])
            self.dataset_predict['y']=y_all
            self.dataset_predict['y_pre']=y_all_pre
            self.r2_all=r2_score(y_true=y_all,y_pred=y_all_pre)
            self.mae_all=mean_absolute_error(y_true=y_all,y_pred=y_all_pre)
            self.rmse_all=root_mean_squared_error(y_true=y_all,y_pred=y_all_pre)
        
        return nn



class MySVR():
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test

    def hyperpara_opt(self):
        record=[]
        # al
        for c in [1e-3,1e-2,1e-1,1,10,100,1e3,1e4,1e5,1e6]:
            for gamma in[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]:
        # solv
        # for c in [1e-4,1e-3,1e-2,]:
        #     for gamma in[1e-8,1e-7,1e-6]:

                svr=SVR(kernel='rbf',C=c,gamma=gamma)
                svr.fit(X=self.X_train,y=self.Y_train)
                y_train_pre=svr.predict(X=self.X_train)
                y_test_pre=svr.predict(X=self.X_test)
                mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=y_train_pre)
                mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=y_test_pre)
                record.append({'C':c,'gamma':gamma,'mae_train':mae_train,'mae_test':mae_test})
        self.hyperpara_mae=pd.DataFrame(record)
        return pd.DataFrame(record)
    
    def determine_hyperpara(self):
        df=self.hyperpara_mae
        self.best_C,self.best_gamma=df.loc[df['mae_test'].idxmin(), ['C', 'gamma']]
        return self.best_C,self.best_gamma

    def train_optmial_model(self):
        c,gamma=self.determine_hyperpara()
        self.optimal_svr=SVR(kernel='rbf',C=c,gamma=gamma)
        self.optimal_svr.fit(X=self.X_train,y=self.Y_train)
        self.y_train_pre=self.optimal_svr.predict(X=self.X_train)
        self.y_test_pre=self.optimal_svr.predict(X=self.X_test)
        self.mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=self.y_test_pre)

        self.r2_train=r2_score(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.r2_test=r2_score(y_true=self.Y_test,y_pred=self.y_test_pre)
        self.rmse_train=root_mean_squared_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.rmse_test=root_mean_squared_error(y_true=self.Y_test,y_pred=self.y_test_pre)
        self.dataset_predict=pd.DataFrame()
        if 1:
            max_len=400
            def pad(y,max_len):
                y=np.pad(y, (0, max_len - len(y)), constant_values=np.nan)
                return y
            self.dataset_predict['y_train']=pad(self.Y_train,max_len)
            self.dataset_predict['y_train_pre']=pad(self.y_train_pre,max_len)
            self.dataset_predict['y_test']=pad(self.Y_test,max_len)
            self.dataset_predict['y_test_pre']=pad(self.y_test_pre,max_len)
        else:
            y_all=np.concatenate([self.Y_train, self.Y_test])
            y_all_pre=np.concatenate([self.y_train_pre, self.y_test_pre])
            self.dataset_predict['y']=y_all
            self.dataset_predict['y_pre']=y_all_pre
            self.r2_all=r2_score(y_true=y_all,y_pred=y_all_pre)
            self.mae_all=mean_absolute_error(y_true=y_all,y_pred=y_all_pre)
            self.rmse_all=root_mean_squared_error(y_true=y_all,y_pred=y_all_pre)
        return self.optimal_svr



class MyRFR():
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test

    def hyperpara_opt(self):
        record=[]
        for n in [2,4,8,16,32,64,128,256]:
            for d in [2,4,6,8,10,12,14,16]:
                rfr=RandomForestRegressor(n_estimators=n,max_depth=d)
                rfr.fit(X=self.X_train,y=self.Y_train)
                y_train_pre=rfr.predict(X=self.X_train)
                y_test_pre=rfr.predict(X=self.X_test)
                mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=y_train_pre)
                mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=y_test_pre)
                record.append({'n_estimators':n,'max_depth':d,'mae_train':mae_train,'mae_test':mae_test})
        self.hyperpara_mae=pd.DataFrame(record)
        return pd.DataFrame(record)
    
    def determine_hyperpara(self):
        df=self.hyperpara_mae
        self.best_n,self.best_depth=df.loc[df['mae_test'].idxmin(), ['n_estimators', 'max_depth']]
        self.best_n,self.best_depth=int(self.best_n),int(self.best_depth)
        return self.best_n,self.best_depth


    def train_optmial_model(self):
        n,d=self.determine_hyperpara()
        self.optimal_rfr=RandomForestRegressor(n_estimators=n,max_depth=d)
        self.optimal_rfr.fit(X=self.X_train,y=self.Y_train)
        self.y_train_pre=self.optimal_rfr.predict(X=self.X_train)
        self.y_test_pre=self.optimal_rfr.predict(X=self.X_test)
        self.mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=self.y_test_pre)

        self.r2_train=r2_score(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.r2_test=r2_score(y_true=self.Y_test,y_pred=self.y_test_pre)
        self.rmse_train=root_mean_squared_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.rmse_test=root_mean_squared_error(y_true=self.Y_test,y_pred=self.y_test_pre)

        self.dataset_predict=pd.DataFrame()
        if 1:
            max_len=400
            def pad(y,max_len):
                y=np.pad(y, (0, max_len - len(y)), constant_values=np.nan)
                return y
            self.dataset_predict['y_train']=pad(self.Y_train,max_len)
            self.dataset_predict['y_train_pre']=pad(self.y_train_pre,max_len)
            self.dataset_predict['y_test']=pad(self.Y_test,max_len)
            self.dataset_predict['y_test_pre']=pad(self.y_test_pre,max_len)
        else:
            y_all=np.concatenate([self.Y_train, self.Y_test])
            y_all_pre=np.concatenate([self.y_train_pre, self.y_test_pre])
            self.dataset_predict['y']=y_all
            self.dataset_predict['y_pre']=y_all_pre
            self.r2_all=r2_score(y_true=y_all,y_pred=y_all_pre)
            self.mae_all=mean_absolute_error(y_true=y_all,y_pred=y_all_pre)
            self.rmse_all=root_mean_squared_error(y_true=y_all,y_pred=y_all_pre)
        return self.optimal_rfr


class MyXGBR():
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train
        self.Y_train=Y_train
        self.X_test=X_test
        self.Y_test=Y_test

    def hyperpara_opt(self):
        record=[]
        # for n in [int(i) for i in range(20,101)]:
        #     for d in [int(i) for i in range(2,20)]:
        for n in [2,4,8,16,32,64,128,256]:
            for d in [2,4,6,8,10,12,14,16]:
                rfr=xgb.XGBRegressor(objective='reg:squarederror',n_estimators=n,max_depth=d)
                rfr.fit(X=self.X_train,y=self.Y_train)
                y_train_pre=rfr.predict(X=self.X_train)
                y_test_pre=rfr.predict(X=self.X_test)
                mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=y_train_pre)
                mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=y_test_pre)
                record.append({'n_estimators':n,'max_depth':d,'mae_train':mae_train,'mae_test':mae_test})
        self.hyperpara_mae=pd.DataFrame(record)
        return pd.DataFrame(record)
    
    def determine_hyperpara(self):
        df=self.hyperpara_mae
        self.best_n,self.best_depth=df.loc[df['mae_test'].idxmin(), ['n_estimators', 'max_depth']]
        self.best_n,self.best_depth=int(self.best_n),int(self.best_depth)
        return self.best_n,self.best_depth


    def train_optmial_model(self):
        n,d=self.determine_hyperpara()
        self.optimal_xgbr=xgb.XGBRegressor(objective='reg:squarederror',n_estimators=n,max_depth=d)
        self.optimal_xgbr.fit(X=self.X_train,y=self.Y_train)
        self.y_train_pre=self.optimal_xgbr.predict(X=self.X_train)
        self.y_test_pre=self.optimal_xgbr.predict(X=self.X_test)
        self.mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=self.y_test_pre)

        self.r2_train=r2_score(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.r2_test=r2_score(y_true=self.Y_test,y_pred=self.y_test_pre)
        self.rmse_train=root_mean_squared_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        self.rmse_test=root_mean_squared_error(y_true=self.Y_test,y_pred=self.y_test_pre)

        self.dataset_predict=pd.DataFrame()
        if 1:
            max_len=400
            def pad(y,max_len):
                y=np.pad(y, (0, max_len - len(y)), constant_values=np.nan)
                return y
            self.dataset_predict['y_train']=pad(self.Y_train,max_len)
            self.dataset_predict['y_train_pre']=pad(self.y_train_pre,max_len)
            self.dataset_predict['y_test']=pad(self.Y_test,max_len)
            self.dataset_predict['y_test_pre']=pad(self.y_test_pre,max_len)
        else:
            y_all=np.concatenate([self.Y_train, self.Y_test])
            y_all_pre=np.concatenate([self.y_train_pre, self.y_test_pre])
            self.dataset_predict['y']=y_all
            self.dataset_predict['y_pre']=y_all_pre
            self.r2_all=r2_score(y_true=y_all,y_pred=y_all_pre)
            self.mae_all=mean_absolute_error(y_true=y_all,y_pred=y_all_pre)
            self.rmse_all=root_mean_squared_error(y_true=y_all,y_pred=y_all_pre)

        return self.optimal_xgbr

class Pareto:
    def __init__(self, df,opt_target:dict):
        self.target_cols=[k for k,v in opt_target.items()]
        self.opt_direction=[v for k,v in opt_target.items()]

        def change_derection(values:list,direction:list):
            return [values[i] if direction[i] == 'min' else -values[i] for i in range(len(values))]
        population = []
        info_cols=[i for i in df.columns if i not in self.target_cols]
        for idx, row in df.iterrows():
            target_values = change_derection([row[col] for col in self.target_cols],self.opt_direction)
            info=[row[i] for i in info_cols]
            # old_idx=row[info_cols[0]]
            # smiles=row[info_cols[1]]
            # cluster=row[info_cols[2]]
            # population.append((idx, target_values,[old_idx,smiles,cluster]))
            population.append((idx, target_values,info))

        self.population = population
        # self.target_cols=target_cols
        self.info_cols=info_cols

    
    def dominate(self, a, b):
        # Defluat opt direction > min()
        a=a[1] # a=(idx,[target1,target2,target3],info_list)
        b=b[1] # b=(idx,[target1,target2,target3],info_list)
        compare= [None]*len(a)
        for i in range(len(a)):  
            if b[i] < a[i]:
                compare[i]='b' # better
            elif b[i] > a[i]:
                compare[i]='w' # worse
            else:
                compare[i]='=' # equal
        if 'b' in compare and 'w' not in compare:
            return True
        else:
            return False

    def pareto_front(self):

        pareto_front = []
        for i in range(len(self.population)):
            is_dominated = False
            for j in range(len(self.population)):
                if i != j and self.dominate(self.population[i], self.population[j]):  # 检查 population[j] 是否支配 population[i]
                    is_dominated = True #如果支配了，[i]必定不在前沿，故break
                    break
            if not is_dominated:
                pareto_front.append(self.population[i])
        df = self.population_to_df(population=pareto_front)
        return df

    def population_to_df(self,population):
        def return_derection(values:list,direction:list):
            return [values[i] if direction[i] == 'min' else -values[i] for i in range(len(values))]
        records = []
        for idx, target_list, info in population:
            row={}
            row.update({k: v for k, v in zip(self.target_cols, return_derection(target_list,direction=self.opt_direction))})
            for i in range(len(info)):
                row[self.info_cols[i]] = info[i]
            # row[self.info_cols[0]] = info[0]
            # row[self.info_cols[1]] = info[1]
            # row[self.info_cols[2]] = info[2]
            records.append(row)
        return pd.DataFrame(records)