import joblib
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import utils,os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error
from sklearn.manifold import TSNE

# Task 1: k-fold cross validation
conf=utils.load_conf()

optimal_rfr_al=joblib.load("./model/al/optimal_rfr.pkl")
optimal_rfr_solv=joblib.load("./model/solv/optimal_rfr.pkl")

df=pd.read_csv(conf.data_path)
feature_cols=conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy

X=df[feature_cols]
Y_al=df["al"]
Y_solv=df["solv"]

X_min,X_max=X.min(),X.max()
Y_al_min,Y_al_max=Y_al.min(),Y_al.max()
Y_solv_min,Y_solv_max=Y_solv.min(),Y_solv.max()
scaled_X=(X-X_min)/(X_max-X_min)
scaled_Y_al=(Y_al-Y_al_min)/(Y_al_max-Y_al_min)
scaled_Y_solv=(Y_solv-Y_solv_min)/(Y_solv_max-Y_solv_min)

def make_k_flod_data(X,Y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = []
    for i,(train_idx, val_idx) in enumerate(kf.split(X=X,y=Y)):
        folds.append((
            X.iloc[train_idx],
            Y.iloc[train_idx],
            X.iloc[val_idx],
            Y.iloc[val_idx]
        ))
    return folds

def train_rfr(X_train,Y_train,n_estimators,max_steps):
    model=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_steps)
    model.fit(X_train,Y_train)
    return model

def k_fold_val(folds,optimal_model,max,min):
    error={"train_mae":[],"test_mae":[]}
    for (train_X,train_Y,test_X,test_Y) in folds:
        model=train_rfr(train_X,train_Y,optimal_model.n_estimators,optimal_model.max_depth)
        y_train_pred=model.predict(X=train_X)
        y_test_pred=model.predict(X=test_X)
        mae_train=mean_absolute_error(y_true=train_Y,y_pred=y_train_pred)
        mae_test=mean_absolute_error(y_true=test_Y,y_pred=y_test_pred)
        error["train_mae"].append(mae_train)
        error["test_mae"].append(mae_test)
    df=pd.DataFrame(error)
    df["train_mae"]=df["train_mae"]*(max-min)
    df["test_mae"]=df["test_mae"]*(max-min)
    return df

def report_error(df):
    print(df)
    train_mean=np.mean(df["train_mae"])
    train_std=np.sqrt(np.var(df["train_mae"]))
    test_mean=np.mean(df["test_mae"])
    test_std=np.sqrt(np.var(df["test_mae"]))
    print(f"train mae {train_mean:.4f}+-{train_std:.4f}")
    print(f"test mae {test_mean:.4f}+-{test_std:.4f}")

if 0:
    al_flod=make_k_flod_data(scaled_X,scaled_Y_al)
    solv_flod=make_k_flod_data(scaled_X,scaled_Y_solv)

    error_al=k_fold_val(al_flod,optimal_rfr_al,Y_al_max,Y_al_min)
    error_solv=k_fold_val(solv_flod,optimal_rfr_solv,Y_solv_max,Y_solv_min)
    report_error(error_al)
    report_error(error_solv)


# Task 2: sampling candidate space and calculate DFT E_ox and G_solv
if 0:
    candidate=pd.read_csv("./outputs/sampling_predict.csv")
    sample_size=round(len(candidate)*0.1)
    assert type(sample_size)==int
    sampled_candidate=candidate.sample(n=sample_size)
    sampled_candidate.to_csv("./data/new_test_set.csv",index=False)

# Task 3: generalization capability test
if 1:
    df1=pd.read_csv("./data/new_test_set.csv")
    df1=df1[["smiles","idx","predict_al","predict_solv"]]
    df1["true_al"]=pd.Series([9999]*len(df1))
    df1["true_solv"]=pd.Series([9999]*len(df1))

    df2=pd.read_csv("./data/result_voltage.csv")
    
    calculated_idx_list=df2["idx"].to_list()
    for idx in calculated_idx_list:
        df1.loc[df1["idx"]==idx,"true_al"]=df2.loc[df2["idx"]==idx,"Voltage"].iloc[0]
        df1.loc[df1["idx"]==idx,"true_solv"]=df2.loc[df2["idx"]==idx,"SolvationEnergy(Hart.)"].iloc[0]
    
    df1=df1[(df1["true_al"]<10)&(df1["true_al"]>0)]
    df1=df1[(df1["true_solv"]<0)]
    df1["predict_solv"]=df1["predict_solv"]
    df1 = df1.reindex(columns=["idx","smiles","true_al","predict_al","true_solv","predict_solv"])
    print(mean_absolute_error(y_true=df1["true_al"],y_pred=df1["predict_al"]))
    print(r2_score(y_true=df1["true_al"],y_pred=df1["predict_al"]))
    print(np.corrcoef(df1["true_al"], df1["predict_al"])[0, 1])
    
    print(mean_absolute_error(y_true=df1["true_solv"],y_pred=df1["predict_solv"]))
    print(r2_score(y_true=df1["true_solv"],y_pred=df1["predict_solv"]))
    print(np.corrcoef(df1["true_solv"], df1["predict_solv"])[0, 1])

    df1.to_csv("./outputs/new_test_set_groudtruth_predict.csv",index=False)


# Task 4: Random sample chemical space -> ramdom_sampling_set, and then compare descriptors' distribution.
if 0:
    if not os.path.exists("./outputs/random_sampling_set.csv"):
        ramdom_sampling_df=utils.CSSampling("./data/data.csv").ramdom_sampling(1024)
        ramdom_sampling_df[feature_cols]=ramdom_sampling_df['smiles'].apply(lambda x:pd.Series(utils.CreateDataset().smiles2descirptors(smiles=x,conf=conf)))
        ramdom_sampling_df.to_csv("./outputs/random_sampling_set.csv",index=False)

    ramdom_sampling_df=pd.read_csv("./outputs/random_sampling_set.csv")
    ramdom_sampling_df=ramdom_sampling_df.dropna()
    tsne_dataset=TSNE(n_components=2).fit_transform(X)
    tsne_chemical_space=TSNE(n_components=2).fit_transform(ramdom_sampling_df[feature_cols])
    pd.DataFrame(tsne_dataset).to_csv("./outputs/t_sne_dataset.csv",index=False)
    pd.DataFrame(tsne_chemical_space).to_csv("./outputs/t_sne_chem_space.csv",index=False)
if 0:
    if not os.path.exists("./outputs/random_sampling_set.csv"):
        ramdom_sampling_df=utils.CSSampling("./data/data.csv").ramdom_sampling(1024)
        ramdom_sampling_df[feature_cols]=ramdom_sampling_df['smiles'].apply(lambda x:pd.Series(utils.CreateDataset().smiles2descirptors(smiles=x,conf=conf)))
        ramdom_sampling_df.to_csv("./outputs/random_sampling_set.csv",index=False)
    from sklearn.decomposition import PCA
    ramdom_sampling_df=pd.read_csv("./outputs/random_sampling_set.csv")
    ramdom_sampling_df=ramdom_sampling_df.dropna()

    umap_dataset=PCA(n_components=2).fit_transform(X)
    umap_chemical_space=PCA(n_components=2).fit_transform(ramdom_sampling_df[feature_cols])
    pd.DataFrame(umap_dataset).to_csv("./outputs/pca_dataset.csv",index=False)
    pd.DataFrame(umap_chemical_space).to_csv("./outputs/pca_chem_space.csv",index=False)
    pass


pass