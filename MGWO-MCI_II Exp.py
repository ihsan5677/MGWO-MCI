import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import random
import time
import matplotlib.pyplot as plt
 
num_features = 21
data_df = pd.read_csv('obs_network.csv')
data_df[data_df.columns[13]].replace('?', np.nan, inplace=True)
data_df.dropna(subset = [data_df.columns[13]], inplace = True)
data_df.iloc[:, 19] = data_df.iloc[:, 19].replace({'B': 0, 'NB': 1,"'P NB'": 2})
data_df.iloc[:, 21] = data_df.iloc[:, 21].replace({'Block': 0, "'NB-No Block'": 1,"'No Block'": 2, 'NB-Wait': 3})
# Split the data into training and testing sets
train_data, test_data = train_test_split(data_df, test_size=0.25)
train_x, test_x = train_data.iloc[:, :num_features], test_data.iloc[:, :num_features]
#this will seperate target variable(label) from both training and testing sets
train_y, test_y = train_data.iloc[:, -1], test_data.iloc[:, -1]

def fitness_function(positions):
    features = np.where(positions>=0.4999)[0]
    train_xf = train_x.iloc[:, features]
    test_xf = test_x.iloc[:, features]
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(train_xf, train_y)
    accuracy = knn_classifier.score(test_xf, test_y)
    alpha = 0.99
    return (alpha*(1-accuracy) + (0.01)*(1-(len(features)/num_features)))

def MGWO_MCI(objf,lb,ub,dim,SearchAgents_no,Max_iter):
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")

    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")

    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")

    #N_pos[] stores the positions of remaining search agents and N_score will store score of each search agents
    N_pos = []
    N_score = []
    #initialize position of the search agents excluding three best agents
    for ii in range(3):
        New_pos = np.zeros(dim)
        New_score = float("inf")
        N_pos.append(New_pos)
        N_score.append(New_score)

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

#Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

#purpose of this array is to keep track of the best fitness value found during each iteration of the main loop.
    Convergence_curve=np.zeros(Max_iter)
 # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    

    timerStart=time.time() 
    k=0
# Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
        
        # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])
        # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
        # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            if (fitness>Alpha_score and fitness<Beta_score ):
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score): 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
               
            #if fitness is greater than the three best agents and less than N_score[k] then store it in N_score and position is store in N_pos[]
            if (fitness>Alpha_score and fitness>Beta_score and fitness>Delta_score and fitness<N_score[k]):
                N_score[k] = fitness
                N_pos[k]=Positions[i,:].copy()
                # increase k by 1 i.e next search agent in N_score or N_pos
                k+=1
                # if k reachs to last agent then again start from the first agent in N_score i.e assign 0 to k
                if(k == 3):
                    k=0  
                    
        a=2-l*((2)/Max_iter);# a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim): 
                       
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
            
                A1=2*a*r1-a # Equation (3.3)
                C1=2*r2 # Equation (3.4)
            
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]) # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha # Equation (3.6)-part 1
                
                r1=random.random()
                r2=random.random()
            
                A2=2*a*r1-a # Equation (3.3)
                C2=2*r2 # Equation (3.4)
            
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]) # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta # Equation (3.6)-part 2       
            
                r1=random.random()
                r2=random.random()
            
                A3=2*a*r1-a # Equation (3.3)
                C3=2*r2 # Equation (3.4)
            
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]) # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta # Equation (3.5)-part 3 
                
                #Crossover operation between new wolves and current wolves
                co1 = np.random.randint(1, len(N_pos[0])-1)
                os1 = np.concatenate((Alpha_pos[:co1], N_pos[0][co1:]))
                #for beta and 2nd omega
                co2 = np.random.randint(1, len(N_pos[1])-1)
                os2 = np.concatenate((Beta_pos[:co2], N_pos[1][co2:]))
                #for delta and 3rd omega
                co3 = np.random.randint(1, len(N_pos[2])-1)
                os3 = np.concatenate((Delta_pos[:co3], N_pos[2][co3:]))
            
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A4=2*a*r1-a; # Equation (3.3)
                C4=2*r2; # Equation (3.4)
                D_os1=abs(C4*os1[j]-Positions[i,j]); # Equation (3.5)-part 1
                X4=os1[j]-A4*D_os1; # Equation (3.6)-part 1
                       
                r1=random.random()
                r2=random.random()
        
                A5=2*a*r1-a; # Equation (3.3)
                C5=2*r2; # Equation (3.4)
            
                D_os2=abs(C5*os2[j]-Positions[i,j]); # Equation (3.5)-part 2
                X5=os2[j]-A5*D_os2; # Equation (3.6)-part 2       

                r1=random.random()
                r2=random.random() 
            
                A6=2*a*r1-a; # Equation (3.3)
                C6=2*r2; # Equation (3.4)
                D_os3=abs(C6*os3[j]-Positions[i,j]); # Equation (3.5)-part 3
                X6=os3[j]-A6*D_os3; # Equation (3.5)-part 3
                #Exploration
                if (1 < A1 < -1) or (A2 > 1 or A2 < -1) or (A3 > 1 or A3 < -1) or (A4 > 1 or A4 < -1) or (A5 > 1 or A5 < -1) or (A6 > 1 or A6 < -1):
                    Positions[i,j] = (X1+X2+X3+X4+X5+X6)/6
                #Exploitation
                else:
                    Positions[i,j] = (X1+X2+X3)/3
                
        Convergence_curve[l]=Alpha_score;

        #if (l%50==0):
           #print(['--------At iteration '+ str(l)+ ' the best fitness is '+ str(Alpha_score)])
    timerEnd=time.time()
    plt.plot(Convergence_curve)
    print('Completed in', (timerEnd - timerStart))
    return Alpha_pos


fit = MGWO_MCI(fitness_function, 0, 1, num_features, 10, 100)
selected_features = np.where(fit>0.5)[0]
print("selected features",selected_features)

mean_fitness = fitness_function(fit)

train_x = train_x.iloc[:, selected_features]
test_x = test_x.iloc[:, selected_features]

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(train_x, train_y)

predicted = knn_classifier.predict(test_x)

accuracy = accuracy_score(test_y, predicted)
    
print("List Accuracy: ", accuracy)
print("List Fitness", mean_fitness)