#https://nbviewer.org/github/srom/distributions/blob/2021-01-05/notebook/NN%20parametrization%20of%20distribution.ipynb
# 3400 data points to train. we may need several a second to hit this scale. Tick data :)
# 10 per second then 10 mins to get 3600
# 10/31/2023 been playng around is this same as pytorchtechnical on github
class DeepNormal(nn.Module):

    def __init__(self, n_inputs, n_hidden):
        super().__init__()

        # Shared parameters
        self.shared_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
        )

        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
        )

        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(n_hidden, 1),
            nn.Softplus(),  # enforces positivity
        )

    def forward(self, x):
        # Shared embedding
        shared = self.shared_layer(x)

        # Parametrization of the mean
        μ = self.mean_layer(shared)

        # Parametrization of the standard deviation
        σ = self.std_layer(shared)

        return torch.distributions.Normal(μ, σ)

def compute_loss(model, x, y):
    normal_dist = model(x)
    neg_log_likelihood = -normal_dist.log_prob(y)
    return torch.mean(neg_log_likelihood)

model.eval()
normal_dist = model(x)    # evaluate model on x with shape (N, M)
mean = normal_dist.mean   # retrieve prediction mean with shape (N,)
std = normal_dist.stddev