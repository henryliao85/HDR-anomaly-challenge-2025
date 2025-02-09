import torch
from supervised_model_train import SimpleBinaryClassifier
import numpy as np
import os

def minmax_scale(errors):
    print(errors)
    err_min = np.min(errors)
    err_max = np.max(errors)
    # 若 max==min，意義是所有誤差都一樣 => score都給0.5或0 
    if err_max == err_min:
        return np.zeros_like(errors)
    return (errors - err_min) / (err_max - err_min)

def normalize_data(X_test):
    # 計算標準差
    stds = np.std(X_test, axis=-1)[:, :, np.newaxis]
    # 以標準差做除法 => 標準化
    X_test = X_test / stds
    # 轉軸
    X_test = np.swapaxes(X_test, 1, 2)
    return X_test

class Model:
    def __init__(self):
        # You could include a constructor to initialize your model here, but all calls will be made to the load method
        in_dim = 100*2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = SimpleBinaryClassifier(in_dim, hidden_dim=128).to(self.device)

    def predict(self, X):
        model = self.clf
        X = torch.flatten(torch.tensor(X).to(self.device, dtype=torch.float32), start_dim=1)
        mu_X = torch.mean(X)
        std_X = torch.std(X)
        errors0 = []
        for batch_x in X:
            batch_x = batch_x.to(self.device, dtype=torch.float32)
            batch_x = (batch_x-mu_X)/std_X
            with torch.no_grad():
                pred = model(batch_x)
                pred = torch.sigmoid(pred)
                # shape = (batch, in_dim)
                #err = model(pred, batch_x)  # shape=(batch,)
            errors0.append(pred.cpu().item())
        return errors0

    def load(self):
        self.clf.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'supervied_model_fft_N_v3-4.pth'), map_location=self.device))

if __name__ == "__main__":
    data = np.load("test_data.npy")
    root_dir = "/home/string-3/Documents/Hackathon/Datasets/"
    dataset0 = normalize_data(np.load(root_dir+"background_test.npz")['data'])
    dataset1 = normalize_data(np.load(root_dir+"bbh_for_challenge_test.npy"))#['data']
    dataset2 = normalize_data(np.load(root_dir+"sglf_for_challenge_test.npy"))#['data']
    
    model = Model()
    model.load()
    #data = np.concatenate((dataset0[:100, :, :], dataset1[:100, :, :], dataset2[:100, :, :]))
    #print(np.concatenate((data[0, :100, :, :], data[1, :100, :, :], data[2, :100, :, :])).shape)
    #prediction = model.predict(data)
    prediction = model.predict(dataset0[:100, :, :])
    print(prediction)
    print(len(prediction))
    import matplotlib.pyplot as plt

    plt.plot(prediction)
    #plt.hist(prediction)
    plt.show()