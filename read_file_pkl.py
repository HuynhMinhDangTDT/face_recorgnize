import pickle
with open('facemodel.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data)