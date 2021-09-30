from hmmlearn import hmm
import numpy as np
import math 
# Definisi Matrik Transisi (states)

states = ('Rainy', 'Sunny')
 
# definisi Matriks Observasi / Matrik emmisi    
observations = ('walk', 'shop', 'clean')

# definisi Matriks Priority
start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
 
transition_probability = {
   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
   }
 
emission_probability = {
   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
   }




model = hmm.MultinomialHMM(n_components=2)
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.7, 0.3],
                            [0.4, 0.6]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                [0.6, 0.3, 0.1]])


array_observasi = np.array([[2,2,0,0,1]])
#menghitung peluang jika observasi {'clean', 'clean',’walk’, ‘walk’, ‘shop’} 
#dimana  0 = walk
#        1 = shop
#        2 = clean
hasil = math.exp(model.score(array_observasi))

print("Peluang : ", hasil)


#mencari state terbaik dari observasi {'clean', 'clean',’walk’, ‘walk’, ‘shop’} 
logprob, seq = model.decode(array_observasi.transpose())
print("Log Probability : ", math.exp(logprob))
print(seq)
