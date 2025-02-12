import torch
from binary_classifier import SimpleBinaryClassifier
import os

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

    def load(self, model_name):
        self.clf.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), model_name), map_location=self.device))

