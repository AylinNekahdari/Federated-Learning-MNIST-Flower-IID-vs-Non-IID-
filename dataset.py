import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 32

def mnist_iid(dataset, num_users):
    num_items = len(dataset) // num_users
    all_indices = np.random.permutation(len(dataset))
    user_groups = {i: set(all_indices[i*num_items:(i+1)*num_items]) for i in range(num_users)}
    return user_groups


def mnist_noniid(dataset, num_users, test=False):
    # Train: 200 shards * 300 images = 60,000
    # Test: 20 shards * 500 images = 10,000
    classes, images = (200, 300) if not test else (20, 500)
    
    # CALCULATE: How many shards per user to use 100% of the data
    shards_per_user = classes // num_users 
    
    classes_indx = list(range(classes))
    users_dict = {i: np.array([], dtype=int) for i in range(num_users)}
    indeces = np.arange(classes * images)
    unsorted_labels = dataset.targets.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :].astype(int)

    for i in range(num_users):
        np.random.seed(i)
        
        if len(classes_indx) < shards_per_user:
            chosen = classes_indx
        else:
            chosen = np.random.choice(classes_indx, shards_per_user, replace=False)
            
        chosen = set(chosen)
        classes_indx = list(set(classes_indx) - chosen)
        
        for t in chosen:
            users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]))

    return users_dict



def mnist_noniid_unequal(dataset, num_users, test=False):
    # Dataset specifics
    classes, images = (1200, 50) if not test else (200, 50)
    classes_indx = list(range(classes))
    users_dict = {i: np.array([], dtype=int) for i in range(num_users)}

    # Prepare sorted indices
    indeces = np.arange(classes * images)
    unsorted_labels = dataset.targets.numpy()
    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :].astype(int)

    # Randomly assign number of classes per user
    min_cls_per_client = 1
    max_cls_per_client = 30
    np.random.seed(42)
    random_selected_classes = np.random.randint(min_cls_per_client, max_cls_per_client + 1, size=num_users)
    random_selected_classes = np.around(random_selected_classes / sum(random_selected_classes) * classes).astype(int)

    # Assign classes to users
    for i in range(num_users):
        class_size = min(random_selected_classes[i], len(classes_indx))
        if class_size == 0:
            continue  # Skip if no classes left
        temp = np.random.choice(classes_indx, class_size, replace=False)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]))
        classes_indx = list(set(classes_indx) - set(temp))

    # If any classes left, assign to user with least samples
    if len(classes_indx) > 0:
        j = min(users_dict, key=lambda x: len(users_dict[x]))
        for t in classes_indx:
            users_dict[j] = np.concatenate((users_dict[j], indeces[t*images:(t+1)*images]))

    return users_dict


class FedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def get_dataloader(dataset, indices=None, shuffle=True):
    if indices is None:
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return DataLoader(FedDataset(dataset, indices), batch_size=BATCH_SIZE, shuffle=shuffle)

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return train, test

