from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor,colorbar,plot


base = pd.read_csv("wines.csv");
x = base.iloc[:,1:14].values;
y = base.iloc[:,0].values;

normalizador = MinMaxScaler(feature_range=(0,1));

x = normalizador.fit_transform(x);


#O x e o y equivale no caso a saída da rede.
#esses valores são denomidados fazendo 5 * raiz da quantidade de registro 
#do dataset, no caso temos 178 registros, fica 5 * sqrt(178) que equivale a 65,65
#ou seja teremos uma matris de 8 por 8 que equivale a 64.
#o input_len se refere a quantidade atributos que o dataset possuí.
som = MiniSom(x = 8,y = 8,input_len=13,sigma=1.0,learning_rate=0.5,random_seed=2);
som.random_weights_init(x);
som.train_random(data=x, num_iteration=100);

pesos = som._weights;
valores_mapa = som._activation_map;

q = som.activation_response(x);


pcolor(som.distance_map().T);
colorbar()

w = som.winner(x[2]);
markers = ['o','s','D'];
color = ['r','g','b'];

#y[y==1] = 0;
#y[y==2] = 1;
#y[y==3] = 2;

for i,j in enumerate(x):
    
    w = som.winner(j);
    plot(
        w[0]+0.5,
        w[1]+0.5,
        markers[y[i]],
        markerfacecolor='None',
        markersize=10,
        markeredgecolor=color[y[i]],
        markeredgewidth=2);