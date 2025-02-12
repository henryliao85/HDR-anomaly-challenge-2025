## Usage

A sample script (e.g., `iharp_sample_notebook.ipynb`) includes two major steps: initial training and making predictions. Below is an example snippet illustrating the workflow:

```python
# Add package directories
sys.path.append('../IHARP_code_submission')
sys.path.append('../IHARP_train')

# Import packages
import sys
import torch
from model import Model
from fit_and_predict import fit, predict

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths and training arguments
checkpoint_path = f"model_checkpoint.pth"

label_dir = "../IHARP_train/data/label"
training_data_dir = "../IHARP_train/data/nc"
predict_data_dir = "../IHARP_train/data/nc_predict"

csv_path = "test.predictions.csv"

custom_args = {'lr': 5e-4, 'num_epochs': 1}  

# Initialize the model (ensure custom_args is defined appropriately)
mdl = Model(args=custom_args)

# ---------------------------
# 1. Initial Training
# ---------------------------
# Train on files from "data/nc" and save the initial checkpoint.
fit(model, checkpoint_path, device, training_data_dir, label_dir)

# ---------------------------
# 3. Prediction
# ---------------------------
# Run predictions on files from "data/nc_validate".
predict(model, device, predict_data_dir, label_dir, csv_path)
```


---

Happy training and predicting!
```

---