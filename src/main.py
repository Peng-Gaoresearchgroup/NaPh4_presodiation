import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")
import yaml,os,sys
import utils
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from model import model
# from model import rfc,pareto
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib,random,sys,torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(42)

def job(job):
    conf=utils.load_conf()
    
    if False:
        dataset_creator=utils.CreateDataset()
        df=dataset_creator.create_dataset(conf=conf)
        df.to_csv(conf.data_path,index=False)
    else:
        df=pd.read_csv(conf.data_path)


    feature_cols=conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy
    
    X=df[feature_cols]
    # Y=df[['al','solv']]
    Y=df[job]

    # scaler=MinMaxScaler()
    X_min,X_max=X.min(),X.max()
    Y_min,Y_max=Y.min(),Y.max()
    scaled_X=(X-X_min)/(X_max-X_min)
    scaled_Y=(Y-Y_min)/(Y_max-Y_min)

    # X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, scaled_Y, test_size=0.3, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=conf.train_test_ratio, random_state=conf.seed)
    
    if 1:
        # train SVR
        svr=model.MySVR(X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test)
        hyperpara_opt=svr.hyperpara_opt()
        hyperpara_opt.to_csv(f'./outputs/{job}/svr_hyperpara_opt.csv',index=False)
        optiaml_svr=svr.train_optmial_model()
        # print(type(svr.y_test_pre))
        svr.dataset_predict.to_csv(f'./outputs/{job}/svr_dataset_predict_r2_{round(svr.r2_train,4)}_{round(svr.r2_test,4)}_mae_{round(svr.mae_train,4)}_{round(svr.mae_test,4)}_rmse_{round(svr.rmse_train,4)}_{round(svr.rmse_test,4)}.csv')
        joblib.dump(optiaml_svr,f'./model/{job}/optimal_svr.pkl')
    if 0:
        # train RFR
        rfr=model.MyRFR(X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test)
        hyperpara_opt=rfr.hyperpara_opt()
        hyperpara_opt.to_csv(f'./outputs/{job}/rfr_hyperpara_opt.csv',index=False)
        optiaml_rfr=rfr.train_optmial_model()
        rfr.dataset_predict.to_csv(f'./outputs/{job}/rfr_dataset_predict_r2_{round(rfr.r2_train,4)}_{round(rfr.r2_test,4)}_mae_{round(rfr.mae_train,4)}_{round(rfr.mae_test,4)}_rmse_{round(rfr.rmse_train,4)}_{round(rfr.rmse_test,4)}.csv')
        joblib.dump(optiaml_rfr,f'./model/{job}/optimal_rfr.pkl')
    if 0:
        # train XGBR
        xgbr=model.MyXGBR(X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test)
        hyperpara_opt=xgbr.hyperpara_opt()
        hyperpara_opt.to_csv(f'./outputs/{job}/xgbr_hyperpara_opt.csv',index=False)
        optiaml_xgbr=xgbr.train_optmial_model()
        xgbr.dataset_predict.to_csv(f'./outputs/{job}/xgbr_dataset_predict_r2_{round(xgbr.r2_train,4)}_{round(xgbr.r2_test,4)}_mae_{round(xgbr.mae_train,4)}_{round(xgbr.mae_test,4)}_rmse_{round(xgbr.rmse_train,4)}_{round(xgbr.rmse_test,4)}.csv')
        joblib.dump(optiaml_xgbr,f'./model/{job}/optimal_xgbr.pkl')
    if 0:
        # train NN
        nnop=model.NNOptimizer(input_dim=len(feature_cols),output_dim=1,batch_size=256,
                        X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test)
        hyperpara_opt=nnop.hyperpara_opt()
        hyperpara_opt.to_csv(f'./outputs/{job}/nn_hyperpara_opt.csv',index=False)


        optimal_nn=nnop.get_optmial_model()
        optimal_nn.train()
        # print(optimal_nn._monitor())
        y_train_pred,y_test_pred=optimal_nn._dataset_predict()
        sta=utils.DatasetStatistician(data_df=conf.data_path,x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test,y_train_pred=y_train_pred,y_test_pred=y_test_pred)
        # print(optimal_nn.depth,optimal_nn.hidden_dim)
        dataset_predict=sta.wrap_train_test()
        r2_train,r2_test,mae_train,mae_test,rmse_train,rmse_test=sta.score_train_test()
        dataset_predict.to_csv(f'./outputs/{job}/nn_dataset_predict_r2_{round(r2_train,4)}_{round(r2_test,4)}_mae_{round(mae_train,4)}_{round(mae_test,4)}_rmse_{round(rmse_train,4)}_{round(rmse_test,4)}.csv')
        # torch.save(optimal_nn,'./model/optimal_nn.pth')

    # Choose best_model
    choose=input('Choose your final model\n(svr,rfr,nn,xgbr):')
    if 'choose' != 'nn':
        best_model=joblib.load(f'./model/{job}/optimal_{choose}.pkl')
    else:
        best_model=torch.load(f'./model/{job}/optimal_{choose}.pth')


    # Chemical space sampling and predict
    css=utils.CSSampling(df=conf.data_path)

    df,optimal_Rs=css.R_analye(target=2.9,optiaml_R_nums=10,y_col=job)
    df.to_csv(f'./outputs/{job}/R_analyze.csv',index=False)
    if 'sampling_predict.csv' not in os.listdir('./outputs/'):
        sampling_df=css.creat_sampling_dataframe(conf=conf,optimal_Rs=optimal_Rs)
        sampling_df[f'predict_{job}']=best_model.predict(X=sampling_df[feature_cols])
    else:
        sampling_df=pd.read_csv('./outputs/sampling_predict.csv')
        sampling_df[f'predict_{job}']=best_model.predict(X=sampling_df[feature_cols])
    sampling_df.to_csv(f'./outputs/sampling_predict.csv',index=False)


def rank(sampling_predict_csv:str):
    ranker=utils.Ranker()
    # df=pd.read_csv(sampling_predict_csv)
    # df=ranker.get_rank_data(df,target_al=2.9)
    # df.to_csv('./outputs/rank_data.csv',index=False)
    df=pd.read_csv('./outputs/rank_data.csv')
    df=ranker.get_rank_data(df,target_al=2.9)

    # df['ranker_al']=abs(df['predict_al']-2.9)
    # df=df[df['ranker_al']<=1]
    # df['ranker_solv']=(-df['predict_solv'])*df['capacity']
    # df['R_nums']=df['smiles'].apply(ranker.get_R_complexity)
    pareto=model.Pareto(df=df,opt_target={'ranker_al':'min',
                                          'ranker_solv':'min',
                                          'sascore':'min',
                                          'scscore':'min',
                                        #   'spatial':'min',
                                            'SASA':'max',
                                          'R_nums':'min'})

    front=pareto.pareto_front()
    # pareto2=model.Pareto(df=front,opt_target={})
    # front=pareto2.pareto_front()
    front=front.sort_values(by='score', ascending=False, inplace=False)
    front.to_csv('./outputs/pareto_fornt.csv',index=False)

    
def main():
    # job('al')
    # job('solv')

    # Rank
    rank(sampling_predict_csv='./outputs/sampling_predict.csv')
main()
