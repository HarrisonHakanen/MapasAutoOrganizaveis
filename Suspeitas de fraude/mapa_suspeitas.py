from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor,colorbar,plot


base = pd.read_csv("credit_data.csv");
base = base.dropna()

base.loc[base.age<0,'age'] = base.age.mean()

X = base.iloc[:,0:4].values
y = base.iloc[:,4].values


normalizador = MinMaxScaler(feature_range=(0,1));

X = normalizador.fit_transform(X);



som = MiniSom(x = 15,y = 15,input_len=4,sigma=1.0,learning_rate=0.5,random_seed=0);
som.random_weights_init(X);
som.train_random(data=X, num_iteration=100);


pcolor(som.distance_map().T);
colorbar()


markers = ['o','s'];
color = ['r','g'];


for i,j in enumerate(X):
    
    w = som.winner(j);
    plot(
        w[0]+0.5,
        w[1]+0.5,
        markers[y[i]],
        markerfacecolor='None',
        markersize=10,
        markeredgecolor=color[y[i]],
        markeredgewidth=2);
    
    
mapeamento = som.win_map(X);
suspeitos = np.concatenate((mapeamento[(4,5)],mapeamento[(6,13)]),axis=0);
suspeitos = normalizador.inverse_transform(suspeitos);



classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        
        if(base.iloc[i,0]) == int(round(suspeitos[j,0])):
            
            classe.append(base.iloc[i,4]);

classe = np.asarray(classe);

suspeitos_final = np.column_stack((suspeitos,classe));
suspeitos_final = suspeitos_final[suspeitos_final[:,4].argsort()]
    
