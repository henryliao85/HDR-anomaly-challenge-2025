import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################################
# 1) UgVgCNN: Processes [B,2,100,160] input (ugosa, vgosa) and produces an embedding.
##############################################################################
class UgVgCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels1=8, out_channels2=8,
                 kernel_size=3, pool_h=5, pool_w=5, out_dim=128):
        """
        A 2-layer CNN that takes input of shape [B, 2, 100, 160] and returns an embedding of shape [B, out_dim].
        Adaptive pooling is set to (pool_h, pool_w) = (5,5).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
        self.lin = nn.Linear(out_channels2 * pool_h * pool_w, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: [B,2,100,160]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)             # [B, out_channels2, pool_h, pool_w]
        x = x.view(x.size(0), -1)      # [B, out_channels2*pool_h*pool_w]
        x = self.lin(x)              # [B, out_dim]
        return x

##############################################################################
# 2) SingleVAEClassifier: A VAE branch that takes an input vector and produces a 1D logit.
##############################################################################
class SingleVAEClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=128):
        """
        A simple VAE-based classifier.
          - Encoder: input_dim -> hidden_dim -> (mu, logvar) (both of size latent_dim)
          - Decoder: latent_dim -> hidden_dim -> output (1 logit)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        out = self.fc_out(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)  # both: [B, latent_dim]
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]
        out = self.decode(z)         # [B, 1]
        return out, mu, logvar

##############################################################################
# 3) MultiVAEClassifier: Contains 12 VAE branches. All receive the same input.
##############################################################################
class MultiVAEClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim=16, hidden_dim=128, num_labels=12):
        super().__init__()
        self.num_labels = num_labels
        self.vae_list = nn.ModuleList([
            SingleVAEClassifier(input_dim, latent_dim, hidden_dim) for _ in range(num_labels)
        ])

    def forward(self, x):
        # x: [B, input_dim]
        outputs = []
        mus = []
        logvars = []
        for vae in self.vae_list:
            out, mu, logvar = vae(x)  # out: [B, 1]
            outputs.append(out)
            mus.append(mu)
            logvars.append(logvar)
        outputs = torch.cat(outputs, dim=1)  # [B, num_labels]
        return outputs, mus, logvars

##############################################################################
# 4) UGosaVGosaSLA_VAE_Model: Overall model combining the CNN and 12 VAE branches.
##############################################################################
class UGosaVGosaSLA_VAE_Model(nn.Module):
    def __init__(self, cnn_args, vae_args):
        """
        Build the model using:
          - A CNN (UgVgCNN) processing the ugosa/vgosa data,
          - Then concatenating its output with the flattened SLA,
          - Then feeding the result into 12 VAE branches (MultiVAEClassifier).
        The input to each VAE branch is of dimension: cnn_out_dim + sla_dim.
        """
        super().__init__()
        self.cnn = UgVgCNN(**cnn_args)
        # Compute input dimension for the VAE branches.
        # We assume sla_dim is provided in vae_args; it should match the flattened SLA size.
        input_dim = cnn_args['out_dim'] + vae_args['sla_dim']
        self.multi_vae = MultiVAEClassifier(input_dim, latent_dim=vae_args.get('latent_dim', 16),
                                             hidden_dim=vae_args.get('hidden_dim', 128),
                                             num_labels=vae_args['num_labels'])

    def forward(self, ugvg, sla):
        """
        ugvg: [B,2,100,160] (ugosa/vgosa from day t-1)
        sla: [B,100,160] (SLA from day t)
        Returns: logits [B, num_labels], and lists of mus and logvars for each branch.
        """
        embed = self.cnn(ugvg)            # [B, cnn_out_dim]
        sla_flat = sla.view(sla.size(0), -1)  # [B, sla_dim]
        x = torch.cat([embed, sla_flat], dim=1)  # [B, cnn_out_dim + sla_dim]
        logits, mus, logvars = self.multi_vae(x)  # logits: [B, num_labels]
        return logits, mus, logvars

##############################################################################
# 5) Minimal Model Class: Wrapper for loading a checkpoint and making predictions.
##############################################################################
class Model:
    def __init__(self, args=None):
        """
        Minimal Model class for prediction.
        Initializes the UGosaVGosaSLA_VAE_Model with default configuration and sets the model to evaluation mode.
        """
        if args is None:
            args = {}
        default_config = {
            'cnn_args': {
                'in_channels': 2,
                'out_channels1': 8,
                'out_channels2': 8,
                'kernel_size': 3,
                'pool_h': 5,      # using (5,5) pooling as requested
                'pool_w': 5,
                'out_dim': 16
            },
            'vae_args': {
                'sla_dim': 16000,   # flattened SLA dimension (100x160)
                'latent_dim': 1,
                'hidden_dim': 8,
                'num_labels': 12
            }
        }
        self.cfg = _merge_dict(default_config, args)
        self.model = UGosaVGosaSLA_VAE_Model(
            cnn_args=self.cfg['cnn_args'],
            vae_args=self.cfg['vae_args']
        )
        self.model.to(device)
        self.model.eval()

    def set_cgf(self, args):
        self.cfg = _merge_dict(self.cfg, args)

    def load(self, checkpoint_path="model_checkpoint.pth"):
        """
        Load model weights from checkpoint using strict=False.
        """
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        result = self.model.load_state_dict(state, strict=False)
        print(f"Loaded model from {checkpoint_path} with strict=False.")
        if result.missing_keys:
            print("Missing keys (new layers randomly initialized):", result.missing_keys)
        if result.unexpected_keys:
            print("Unexpected keys:", result.unexpected_keys)

    def predict(self, ugvg, sla):
        """
        Make a prediction using the loaded model.
        Parameters:
            ugvg: Tensor of shape [B,2,100,160] representing ugosa and vgosa data.
            sla: Tensor of shape [B,100,160] representing SLA data.
        Returns:
            probs: Tensor of shape [B, num_labels] with prediction probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(ugvg, sla)  # [B,12]
            probs = torch.sigmoid(logits)
        return probs

def _merge_dict(base, override):
    """
    Helper function to merge two dictionaries.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            tmp = dict(base[k])
            tmp.update(v)
            out[k] = tmp
        else:
            out[k] = v
    return out
