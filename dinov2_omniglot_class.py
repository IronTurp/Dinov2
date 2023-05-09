import torch
from torchvision.datasets import Omniglot
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

class DINOClassifier:
    def __init__(self, dinov2_version='vitl14', root="./omniglot", batch_size=64, test_size=0.3, random_state=42):
        self.root = root
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(), 
            transforms.Resize((98, 98), antialias=True)
        ])
        self.dataloader = None
        self.dinov2 = self.pick_dinov2_version(dinov2_version)
        self.dinov2 = self.dinov2.to(self.device)
        self.model = None

    def pick_dinov2_version(self, version):
        if version == 'vits14':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        elif version == 'vitl14':
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()
        else:
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()

    def load_data(self):
        dataset = Omniglot(
            root=self.root, download=True, transform=self.transform
        )
        self.dataloader = DataLoader(
            dataset, shuffle=True, batch_size=self.batch_size
        )

    def extract_embeddings(self):
        all_embeddings, all_targets = [], []
        with torch.no_grad():
            for images, targets in tqdm(self.dataloader):
                images = images.to(self.device)
                embedding = self.dinov2(images)
                all_embeddings.append(embedding)
                all_targets.append(targets)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        return all_embeddings.cpu().numpy(), all_targets.cpu().numpy()

    def train(self, X_train, y_train):
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        test_acc = self.model.score(X_test, y_test)
        return test_acc

# Usage example:
classifier = DINOClassifier()
classifier.load_data()
X, y = classifier.extract_embeddings()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
classifier.train(X_train, y_train)
test_acc = classifier.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
