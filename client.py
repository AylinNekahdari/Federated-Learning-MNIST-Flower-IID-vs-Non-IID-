import torch
import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.app import Context
from model import Net
from dataset import load_mnist, mnist_iid, mnist_noniid, mnist_noniid_unequal, get_dataloader
from train import train, test

DEVICE = torch.device("cpu")


# --- Configuration ---
NUM_CLIENTS = 10
# Options: "iid", "non-iid-equal", "non-iid-unequal"
DISTRIBUTION_TYPE = "non-iid-equal" 

# 1. Load the data first
train_ds, test_ds = load_mnist()

# 2. Select partitioning based on DISTRIBUTION_TYPE
if DISTRIBUTION_TYPE == "iid":
    print(f"Partitioning data in IID fashion for {NUM_CLIENTS} clients...")
    client_splits = mnist_iid(train_ds, NUM_CLIENTS)

elif DISTRIBUTION_TYPE == "non-iid-equal":
    print(f"Partitioning data in Non-IID (Equal) fashion...")
    client_splits = mnist_noniid(train_ds, NUM_CLIENTS)

elif DISTRIBUTION_TYPE == "non-iid-unequal":
    print(f"Partitioning data in Non-IID (Unequal) fashion...")
    client_splits = mnist_noniid_unequal(train_ds, NUM_CLIENTS)

else:
    raise ValueError(f"Unknown distribution type: {DISTRIBUTION_TYPE}")


class MNISTClient(NumPyClient):
    def __init__(self, cid):
        self.model = Net().to(DEVICE)
        self.trainloader = get_dataloader(train_ds, client_splits[int(cid)])
        self.testloader = get_dataloader(test_ds, shuffle=False)

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                               [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, DEVICE)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = test(self.model, self.testloader, DEVICE)
        return loss, len(self.testloader.dataset), {"accuracy": acc}

def client_fn(cid: str):
    return MNISTClient(cid)

app = ClientApp(client_fn=client_fn)

