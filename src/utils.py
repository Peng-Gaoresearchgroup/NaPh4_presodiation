from rdkit import Chem
from rdkit.Chem import Descriptors,AllChem,Descriptors3D
import pandas as pd
import time,yaml
import numpy as np
import torch
import random,json
from sklearn.metrics import mean_absolute_error,r2_score,root_mean_squared_error
import requests as rq
from rdkit.Chem import rdFreeSASA
random.seed= 42
def log(func):    
    def wrapper(*args, **kwargs):
        t=time.strftime("%H:%M:%S", time.localtime())
        print(f"[INFO {t}] 调用 {func.__name__}()")
        if func.__name__=='to_csv':
            print(f'               {args[1]} 已保存')

        return func(*args, **kwargs)
    return wrapper

@log
def load_conf():
    with open("./conf/conf.yaml") as f:
        conf = SafeDict(yaml.safe_load(f))
    return conf

pd.DataFrame.to_csv=log(pd.DataFrame.to_csv)

class CreateDataset():
    '''
    对smiles集合进行预处理，产生可用于模型训练的数据。
    '''
    def __init__(self):
        pass
    def get_ion(self,smiles:str):
        '''
        从Pubchem smiles数据集中提取含B-阴离子部分。
        '''
        smiles = smiles.strip()
        lst=[i for i in smiles.split('.') if 'B' in i and '+' not in i and '-' in i]
        if len(lst)==1:
            return lst[0]
        else:
            return None
        
    def wash(self,smiles):
        '''
        清洗。
        '''
        try:
            mol=Chem.MolFromSmiles(smiles)
            mol=Chem.AddHs(mol)
            # 去除同位素
            tmp=[a.SetIsotope(0) for a in mol.GetAtoms()]
            mol=Chem.RemoveHs(mol)

            # 去除特定元素，Br Cl对集流体不好，Se Si一般不好合成
            if any(a.GetSymbol() in ['Br', 'Cl', 'Se', 'Si'] for a in mol.GetAtoms()):
                return None
            else:
                #去除活泼氢
                n=[]
                h_atoms=[a for a in mol.GetAtoms() if a.GetSymbol()=='H']
                for a in h_atoms:
                    n+=a.GetNeighbors()
                if 'S' in n or 'O' in n or 'N' in n:
                    return None
                else:
                    # 去除手性
                    return Chem.MolToSmiles(mol,isomericSmiles=False)
        except:
            return None
    
    def filter_by_capacity(self,smiles:str,lower_limit:float):
        '''
        通过理论容量筛选分子。
        '''
        # print(f'filter_by_capacity: {smiles}')
        mol=Chem.MolFromSmiles(smiles)
        wt=Descriptors.MolWt(mol)
        capa=(1*96500)/(3.6*(wt+23))
        if capa>=lower_limit:
            return smiles
        else:
            return None
    
    def smiles2descirptors(self,smiles,conf): 
        #desdic is a dic containing what descriptors we choose,{_get:['a','b'],_2d:['c']......}, see conf.yaml.
        '''Input a SMILES string, return its descriptors' pd.series calculated by rdkit'''
        seed=conf.seed
        desdic=conf.descriptors
        print(f'Processing {smiles}')    
        mol=Chem.MolFromSmiles(smiles) 
        if mol is None:
            raise ValueError("SMILES Not valid")
        try:
        # if True:
            # Get 2D descriptors
            des={f'{i}': getattr(mol, f'Get{i}')() for i in desdic._get}
            des.update({f'{i}': getattr(Descriptors, f'{i}')(mol) for i in desdic._2d})
            # 3D Embed
            AllChem.EmbedMolecule(mol, useRandomCoords=True,maxAttempts=5000, randomSeed=seed)
            AllChem.MMFFOptimizeMolecule(mol)
            mol=Chem.AddHs(mol)
            params = AllChem.ETKDG()
            params.randomSeed = seed
            AllChem.EmbedMolecule(mol, params)
            # Get 3D descriptors
            des.update({f'{i}': getattr(Descriptors3D, f'{i}')(mol) for i in desdic._3d})
            #Get Diy descriptors
            def SP2ratio(mol):
                total_c=0
                sp2=0
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum()==6:
                        total_c+=1
                        hybridization=atom.GetHybridization()
                        if hybridization==Chem.HybridizationType.SP2:
                            sp2+=1
                sp2_frac=sp2/total_c
                return sp2_frac
            des.update({desdic._diy[0]:SP2ratio(mol)})

            def find_negatively_charged_atoms(mol):
                try:
                    AllChem.EmbedMolecule(mol, useRandomCoords=True,maxAttempts=5000, randomSeed=seed)
                    AllChem.MMFFOptimizeMolecule(mol)
                    mol=Chem.AddHs(mol)
                    params = AllChem.ETKDG()
                    params.randomSeed = seed
                    AllChem.EmbedMolecule(mol, params)
                    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
                    ewg=['F','H','B','O']
                    edg=['S','C','N','P','Si']
                    # edg=['S','O','N']
                    edg_num=0
                    ewg_num=0
                    # AllChem.ComputeGasteigerCharges(mol)  # 计算Gasteiger部分电荷

                    charges = [atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()]
                    # charges = [atom.GetDoubleProp('molAtomMapNumber') for atom in mol.GetAtoms()]
                    # max_charge = max(charges)
                    min_charge = min(charges)
                    min_charge_atom = None
                    
                    for atom in mol.GetAtoms():
                        if atom.GetDoubleProp('_GasteigerCharge') == min_charge:
                            # min_charge_atom = atom.GetSymbol()
                            min_charge_atom = atom
                            break
                    # print(min_charge_atom.GetSymbol())
                    n=[neighbor for neighbor in min_charge_atom.GetNeighbors()]
                    n_charge=[i.GetDoubleProp('_GasteigerCharge') for i in n]
                    n_min_charge=min(n_charge)

                    for i in n:
                        # print(i.GetSymbol())
                        if i.GetSymbol() in edg:
                                edg_num+=1
                        elif i.GetSymbol() in ewg:
                                ewg_num+=1
                        # for j in [neighbor.GetSymbol() for neighbor in i.GetNeighbors()]:
                        #     # print(j)
                        #     if j in edg:
                        #         edg_num+=1
                        #     elif j in ewg:
                        #         ewg_num+=1
                    return min_charge_atom.GetAtomicNum(),edg_num/len(n),n_min_charge
                except:
                    return None,None,None
            a,b,c=find_negatively_charged_atoms(mol)
            des.update({desdic._diy[1]:a})
            des.update({desdic._diy[2]:b})
            des.update({desdic._diy[3]:c})

            def element_ratio(mol):
                elements=['C','H','O','N','F','S','B','P']
                atoms=[i.GetSymbol() for i in mol.GetAtoms()]
                ratio=1/len(atoms)
                d={}
                for atom in elements:
                    d.update({atom:0})
                for atom in atoms:
                    if atom in elements:
                        d[atom]+=ratio

                return d['C'],d['H'],d['O'],d['N'],d['F'],d['S'],d['B'],d['P']
            c,h,o,n,f,s,b,p=element_ratio(mol)
            des.update({desdic._diy[4]:c})
            des.update({desdic._diy[5]:h})
            des.update({desdic._diy[6]:o})
            des.update({desdic._diy[7]:n})
            des.update({desdic._diy[8]:f})
            des.update({desdic._diy[9]:s})
            des.update({desdic._diy[10]:b})
            des.update({desdic._diy[11]:p})
            # des.update({desdic._diy[12]:se})
            # des.update({desdic._diy[13]:p})

                
            print(f'{smiles} Processing done')
            return des

        except:
        # else:
            print(f'{smiles} try Error')
            des={f'{i}': float('nan') for i in desdic._get}
            des.update({f'{i}': float('nan') for i in desdic._2d})
            des.update({f'{i}': float('nan') for i in desdic._3d})
            des.update({f'{i}': float('nan') for i in desdic._diy})
            return des
    
    def read_dft(self,smiles:str,dft_csv:str):
        try:
            df=pd.read_csv(dft_csv)
            al=df[df['smiles']==smiles]['Voltage'].values[0]
            solv=df[df['smiles']==smiles]['SolvationEnergy(Hart.)'].values[0]
            if al>=6.0 or al<=-0.33:
                al=None
            if solv==0.0:
                solv==None
            return al,solv
        except:
            return None,None
    def create_dataset(self,conf):
        '''
        主入口，执行全部操作，输出dataset.
        '''
        df=pd.read_csv(conf.pubchem_csv)
        data=pd.DataFrame()
        data['smiles']=df['SMILES'].copy()

        # Get anion
        data['smiles']=data['smiles'].apply(lambda x: self.wash(self.get_ion(x)))
        data=data.dropna()
        data['smiles']=data['smiles'].apply(lambda x: self.filter_by_capacity(smiles=x,lower_limit=75))
        data=data.dropna()
        # Get labels
        data[['al','solv']]=data['smiles'].apply(lambda x:pd.Series(self.read_dft(smiles=x,dft_csv=conf.dft_csv)))
        
        # Ger features
        data[conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy]=data['smiles'].apply(lambda x:pd.Series(self.smiles2descirptors(smiles=x,conf=conf)))

        # duplicates
        data=data.drop_duplicates(subset=['smiles'])
        data=data.dropna()
        data=data.reset_index()
        data['idx']=data.index
        return data
        
class DatasetStatistician():
    '''
    Statistic dataset, for paper analyze.
    '''
    def __init__(self,data_df,x_train,x_test,y_train,y_test,y_train_pred,y_test_pred):
        self.df=data_df
        self.X_train=x_train
        self.X_test=x_test
        self.Y_train=y_train
        self.Y_test=y_test
        self.y_train_pre=y_train_pred
        self.y_test_pre=y_test_pred
        pass
    def element_desrtibution(self):
        from collections import Counter
        df=self.df
        li=df['smiles'].to_list()
        # li=list(set(li))
        # print(len(li))
        elements=[]
        for smiles in li:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            if mol:
                for atom in mol.GetAtoms():
                    elements.append(atom.GetSymbol())
        # elements=tuple(elements)
        counter=Counter(elements)
        return counter
    def wrap_train_test(self):
        df=pd.DataFrame()
        max_len=400
        def pad(y,max_len):
            y=np.pad(y, (0, max_len - len(y)), constant_values=np.nan)
            return y
        # print(self.Y_train.shape)
        df['y_train']=pad(self.Y_train,max_len)
        df['y_train_pre']=pad(self.y_train_pre,max_len)
        df['y_test']=pad(self.Y_test,max_len)
        df['y_test_pre']=pad(self.y_test_pre,max_len)
        return df
    def score_train_test(self):
        mae_train=mean_absolute_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        mae_test=mean_absolute_error(y_true=self.Y_test,y_pred=self.y_test_pre)
        r2_train=r2_score(y_true=self.Y_train,y_pred=self.y_train_pre)
        r2_test=r2_score(y_true=self.Y_test,y_pred=self.y_test_pre)
        rmse_train=root_mean_squared_error(y_true=self.Y_train,y_pred=self.y_train_pre)
        rmse_test=root_mean_squared_error(y_true=self.Y_test,y_pred=self.y_test_pre)
        return r2_train,r2_test,mae_train,mae_test,rmse_train,rmse_test

class SafeDict(dict):
    '''
    Convert how the value is indexed.
    '''

    def __getattr__(self, name):
        value = self.get(name, None) 
        if isinstance(value, dict):
            return SafeDict(value)
        elif isinstance(value, list):
            return [SafeDict(item) if isinstance(item, dict) else item for item in value]
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __dir__(self):
        return super().__dir__() + list(self.keys())

class CSSampling():
    '''Analyze R groups of smiles set, make sample method, avoid useless screen.'''
    def __init__(self,df:str):
        self.df=pd.read_csv(df)
    def get_R(self,smiles:str):
        '''
        Smiles should be like: R1[B-](R2)(R3)R4, only one B.
        Return (R1,R2,R3,R4),with "*" as connection mark.
        '''
        # print(f'get_R :{smiles}')
        if 'B' in smiles:
            try:
                empty_atom=Chem.MolFromSmiles('[*]').GetAtoms()[0]
                # print(empty_atom.GetSymbol())

                mol=Chem.MolFromSmiles(smiles)
                mol=Chem.AddHs(mol)
                rwmol=Chem.RWMol(mol)
                b_atom = [atom for atom in rwmol.GetAtoms() if atom.GetSymbol() == 'B'][0]
                neibors=[i for i in b_atom.GetNeighbors()]
                for n in neibors:
                    rwmol.RemoveBond(b_atom.GetIdx(), n.GetIdx())
                    rwmol.AddAtom(empty_atom)
                    mark_atoms = [atom for atom in rwmol.GetAtoms() if atom.GetSymbol() == '*']
                    for i in mark_atoms:
                        if i.GetDegree()==0:
                            rwmol.AddBond(n.GetIdx(),i.GetIdx(),Chem.BondType.SINGLE)
                product=Chem.MolToSmiles(rwmol)

                Rs=[i for i in product.split('.') if i !='[B-]']
                #去除过大官能团
                if any(Descriptors.MolWt(Chem.MolFromSmiles(i))>=500 for i in Rs):
                    # return None,None,None,None 
                    return  float('nan'),float('nan'),float('nan'),float('nan')
                else:
                    Rs=[Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(i))) for i in Rs]
                    Rs.sort(key=len)                   
                    return Rs[0],Rs[1],Rs[2],Rs[3]
            except:
                # return None,None,None,None
                return float('nan'),float('nan'),float('nan'),float('nan')
        else:
            # return None,None,None,None
            return float('nan'),float('nan'),float('nan'),float('nan')
        
    def assemble(self,r1,r2,r3,r4):
        '''Input R1,R2,R3,R4, return smiles of [B-]R1,R2,R3,R4.'''
        # print(r1,r2,r3,r4)
        mol=Chem.MolFromSmiles(f'[B-].{r1}.{r2}.{r3}.{r4}')
        rw=Chem.RWMol(mol)
        mark_site_pairs=[] # mark_idx : site_atom_idx
        B_idx=None
        for a in rw.GetAtoms():
            if a.GetSymbol()=='*':
                for n in a.GetNeighbors():
                        mark_site_pairs.append((a.GetIdx(),n.GetIdx()))
            elif a.GetSymbol()=='B':
                B_idx=a.GetIdx()
        # print(mark_site_pairs,B_idx)

        # Reset Bond
        for pair in mark_site_pairs:
            rw.RemoveBond(pair[0],pair[1])
            rw.AddBond(B_idx,pair[1],Chem.rdchem.BondType.SINGLE)
        
        # Remove *
        for i in range(4):
            for a in rw.GetAtoms():
                if a.GetSymbol()=='*':
                    rw.RemoveAtom(a.GetIdx())
                    break

        return Chem.MolToSmiles(rw)
    
    def collect_R_set(self):
        '''
        收集R基团，形成set.
        '''
        df=self.df

        li=df['smiles'].to_list()
        Rs=[]
        for smiles in li:
            Rs+=list(self.get_R(smiles=smiles))
        R_set=set([i for i in Rs if 'B' not in i])
        self.R_set=R_set
        self.R_dict={i: element for i, element in enumerate(R_set)}
        # self.R_num=len(self.R_set)

    def creat_sampling_dataframe(self,conf,optimal_Rs):
        
        smiles_list=[]
        for r1 in optimal_Rs:
            for r2 in optimal_Rs:
                for r3 in optimal_Rs:
                    for r4 in optimal_Rs:
                        smiles=self.assemble(r1,r2,r3,r4)
                        if smiles not in smiles_list:
                            smiles_list.append(smiles)
        # record=[]

                
        df=pd.DataFrame()
        df['smiles']=smiles_list
        df['idx']=df.index
        cd=CreateDataset()
        df[conf.descriptors._get+conf.descriptors._2d+conf.descriptors._3d+conf.descriptors._diy]=df['smiles'].apply(lambda x:pd.Series(cd.smiles2descirptors(smiles=x,conf=conf)))
        return df
    
    def R_analye(self,target,optiaml_R_nums,y_col):
        self.collect_R_set()
        smiles_y=dict(zip(self.df['smiles'].to_list(),self.df[y_col].to_list()))
        result_dict={}
        d=[]
        for r in self.R_set:
            y=[]
            for k,v in smiles_y.items():
                if r in self.get_R(smiles=k):
                    y.append(v)
            result_dict.update({r:y})

        # result_dict = dataset_df.groupby('R1')['predict_y'].apply(list).to_dict()
        for k,v in result_dict.items():
            mean,var,num=np.mean(v),np.var(v),len(v)
            d.append({'R1':k,'mean':mean,'var':var,'num':num})
        # print(pd.DataFrame(d))
        df=pd.DataFrame(d)
        df['difference'] = abs(df['mean'] - target)+ df['var'] - df['num']
        top_n_rows = df.nsmallest(optiaml_R_nums, 'difference')
        optimal_Rs = top_n_rows['R1'].tolist()

        return df,optimal_Rs
    def ramdom_sampling(self,sample_num):
        self.collect_R_set()
        R_li=list(self.R_set)
        _,optimal_Rs=self.R_analye(target=2.9,optiaml_R_nums=10,y_col="al")
        smiles_list=[]
        for _ in range(sample_num):
            r1=random.choice(R_li)
            r2=random.choice(R_li)
            r3=random.choice(R_li)
            r4=random.choice(R_li)
            smiles_list.append(self.assemble(r1,r2,r3,r4))
        return pd.DataFrame({"smiles":smiles_list})

class Ranker():
    def get_scscore(self,smiles,tar=['sascore','scscore','spatial']):
        time.sleep(0.15)
        api_url = 'https://askcos.mit.edu/api/molecular-complexity/call-async'
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0',
                "Accept": "application/json",
                "Content-Type": "application/json"
                }
        request_data = {"smiles": smiles,"complexity_metrics": tar}
        print(f'SCScore: Processing {smiles}')
        re1 = rq.post(api_url, json=request_data,headers=headers)
        if re1.status_code != 200:
            print(f"Post failed: {re1.status_code}")
            return [None for i in tar]
        re2=rq.get(f'https://askcos.mit.edu/api/legacy/celery/task/{re1.json()}')
        if re2.status_code != 200:
            print(f"Get failed: {re2.status_code}")
            return [None for i in tar]
        print(f"Post sucess: {re2.status_code}")
        data=json.loads(re2.text)
        state = data["state"]
        result =data["output"]["result"]
        if state != "SUCCESS":
            print(f"Failed: {re2.status_code}")
            return [None for i in tar]
        print(f"Sucess: {re2.status_code}")
        return [float(result[i]) for i in tar]
                     
    def get_specific_capacity(self,smiles):
        mol=Chem.MolFromSmiles(smiles)
        mol=Chem.AddHs(mol)
        mol_weight = Descriptors.MolWt(mol)
        Na_num = 1
        specific_capacity=Na_num*96500/(3.6*mol_weight)
        return specific_capacity

    def get_R_complexity(self,smiles):
            css=CSSampling(df='./data/data.csv')
            r1,r2,r3,r4=css.get_R(smiles)
            if float('nan')  in [r1,r2,r3,r4]:
                return 5
            return len(set([r1,r2,r3,r4]))
    def get_all_elements(self,path):
        df=pd.read_csv(path)
        smiles_li=df['smiles'].to_list()
        result=[]
        for smlies in smiles_li:
            mol=Chem.MolFromSmiles(smlies)
            for a in mol.GetAtoms():
                if a.GetSymbol() not in result:
                    result.append(a.GetSymbol())
        return result
    

    def get_SASA(self,smiles):
        atom_radii={'H':1.2,
                    'C':1.7,
                    'B':1.74,
                    'N':1.5,
                    'F':1.35,
                    'O':1.4,
                    'S':1.85,
                    'P':1.63}# http://doi.org/10.1023/A:1011625728803
        mol=Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol, useRandomCoords=True,maxAttempts=5000, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol=Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)

        area=rdFreeSASA.CalcSASA(mol,radii=[atom_radii[a.GetSymbol()] for a in mol.GetAtoms()])
        return area
        
    def get_rank_data(self,df,target_al):
        if 'sascore' not in df.columns:
            df[['sascore','scscore','spatial']]=df['smiles'].apply(lambda x: pd.Series(self.get_scscore(x,['sascore','scscore','spatial'])))
        df['capacity']=df['smiles'].apply(self.get_specific_capacity)
        df['ranker_al']=abs(df['predict_al']-target_al)
        df['ranker_solv']=(-df['predict_solv'])*df['capacity']
        df['R_nums']=df['smiles'].apply(self.get_R_complexity)
        df['SASA']=df['smiles'].apply(self.get_SASA)
        df=df[df['ranker_al']<=0.5]
        df=df[df['R_nums']<=2]

        columns_to_normalize=['capacity','sascore','scscore','ranker_al','predict_solv','R_nums','SASA']
        copy=df.copy()
        copy[columns_to_normalize] = copy[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        df['score']=copy['capacity']+copy['SASA']-copy['sascore']-copy['scscore']-copy['ranker_al']-copy['predict_solv']-copy['R_nums']
        return df
    



