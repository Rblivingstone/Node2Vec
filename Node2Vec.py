from keras.layers import Dense,Input
from keras.models import Model
from keras import optimizers
import pandas as pd
import pyodbc as db
import networkx as nx
import numpy as np

class Node2Vec():
    
    
    def __init__(self,graph,hidden_units = 2):
        self.G=graph
        self.node_dict = {list(self.G.nodes)[i]:i for i in range(len(list(self.G.nodes)))}
        self.build_model(hidden_units)
        return
    
    def generate_random_walks(self,walklen=10,numwalks=5):
        walks = []
        for startnode in list(self.G.nodes):
            if len(walks)%1000==0:
                print('Progress Generating Random Walks: {0}%'.format(100*(len(walks)/(numwalks*len(list(self.G.nodes))))))
            for j in range(numwalks):
                walk = [startnode]
                for i in range(walklen-1):
                    obj = next(nx.bfs_successors(self.G,walk[-1]))[1]
                    if len(obj)!=0:
                        walk.append(np.random.choice(obj))
                    else:
                        walk.append(walk[-1])
                walks.append(walk)
        
        
        walks2 = np.vectorize(self.node_dict.get)(np.array(walks))
        return walks2
    
    def generate_skip_grams(self,walks):
        X = []
        y1,y2,y3,y4 = [],[],[],[]
        for i in range(len(walks)):
            if i%1000==0:
                print('Progress Generating Skip-Grams: {0}%'.format(100*(i/len(walks))))
            temp_X = [0] * len(self.node_dict)
            temp_y1 = [0] * len(self.node_dict)
            temp_y2 = [0] * len(self.node_dict)
            temp_y3 = [0] * len(self.node_dict)
            temp_y4 = [0] * len(self.node_dict)
            for j in range(len(walks[i])):
                if j>0:
                    temp_y1[walks[i][j-1]] = 1
                if j>1:
                    temp_y2[walks[i][j-2]] = 1
                if j<9:
                    temp_y3[walks[i][j+2]] = 1
                if j<8:
                    temp_y4[walks[i][j+1]] = 1
                temp_X[walks[i][j]] = 1
                X.append(temp_X)
                y1.append(temp_y1)
                y2.append(temp_y2)
                y3.append(temp_y3)
                y4.append(temp_y4)
        return np.array(X),np.array(y1),np.array(y2),np.array(y3),np.array(y4)
    
    def build_model(self,hidden_units=2):
        input_nodes = Input(shape=(len(self.node_dict),))
        hidden_layer = Dense(hidden_units)(input_nodes)
        output_nodes1 = Dense(len(self.node_dict))(hidden_layer)
        output_nodes2 = Dense(len(self.node_dict))(hidden_layer)
        output_nodes3 = Dense(len(self.node_dict))(hidden_layer)
        output_nodes4 = Dense(len(self.node_dict))(hidden_layer)
        self.train_model = Model(input_nodes,[output_nodes1,output_nodes2,output_nodes3,output_nodes4])
        self.vector_model = Model(input_nodes,hidden_layer)
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.09, nesterov=True)
        self.train_model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['acc'])
        
    def fit(self,epochs = 100):
        walks = self.generate_random_walks()
        X,y1,y2,y3,y4 = self.generate_skip_grams(walks)
        self.train_model.fit(X,[y1,y2,y3,y4],epochs=epochs)
        
    def predict(self,node_list):
        X = []
        n = np.vectorize(self.node_dict.get)(np.array(node_list))
        for i in range(len(n)):
            temp_X = [0] * len(self.node_dict)
            temp_X[n[i]] = 1
            X.append(temp_X)
        X = np.array(X)
        return self.vector_model.predict(X)
