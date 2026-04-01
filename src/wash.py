import pandas as pd
import random
import pandas as pd 
from utils import CreateDataset,load_conf
random.seed= 42


conf = load_conf()
df=pd.read_csv(conf.pubchem_csv)
creator=CreateDataset()
df2=creator.create_dataset(conf)
df2.to_csv("./data/data.csv",index=False)