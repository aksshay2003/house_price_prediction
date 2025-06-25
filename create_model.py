import pickle

with open("models/model_pickle_production.pkl", "wb") as f:
    pickle.dump(trained_model, f)
