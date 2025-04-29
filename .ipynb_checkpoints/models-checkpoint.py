from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
IMG_WIDTH = 224
IMG_HEIGHT = 224

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

def to_tensor(path):
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    return tensor.to(device)

class ModelWrapper(nn.Module):
    def __init__(self, model_class, weights_path, *model_args, **model_kwargs):
        super(ModelWrapper, self).__init__()
        self.device = device
        self.model = model_class(*model_args, **model_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)


class ResNet18(nn.Module):
    def __init__(self, num_classes, name):
        super(ResNet18, self).__init__()
        self.name = name
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for name, param in self.base_model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

        self.base_model.fc = nn.Sequential(nn.Linear(self.base_model.fc.in_features, 512),
                                           nn.ReLU(),
                                           nn.Dropout(),
                                           nn.Linear(512, num_classes))

    def forward(self, x):
        return self.base_model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred = self.forward(x)
            predicted_class = torch.argmax(pred, dim=1)
        return predicted_class


class ResNet34(nn.Module):
    def __init__(self, num_classes, name):
        super(ResNet34, self).__init__()
        self.name = name
        # Load pretrained ResNet34
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Freeze feature extractor layers
        for name, param in self.base_model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

        self.base_model.fc = nn.Sequential(nn.Linear(self.base_model.fc.in_features, 512),
                                           nn.ReLU(),
                                           nn.Dropout(),
                                           nn.Linear(512, num_classes))

    def forward(self, x):
        return self.base_model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred = self.forward(x)
            predicted_class = torch.argmax(pred, dim=1)
        return predicted_class


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        predictions = [model(x).data for model in self.models]
        avg_predictions = torch.mean(torch.stack(predictions), dim=0)
        return avg_predictions

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            predicted_class = torch.argmax(predictions, dim=1)
        return predicted_class

class BreedExtractor:
    def __init__(self, dog_weights_paths, cat_weights_paths):

        # Загружаем и оборачиваем в ансамбль
        self.dog_model = EnsembleModel([
            self._load_model(ResNet18, dog_weights_paths[0], num_classes=25, name='DogBreed'),
            self._load_model(ResNet34, dog_weights_paths[1], num_classes=25, name='DogBreed')
        ]).to(device)

        self.cat_model = EnsembleModel([
            self._load_model(ResNet18, cat_weights_paths[0], num_classes=12, name='CatBreed'),
            self._load_model(ResNet34, cat_weights_paths[1], num_classes=12, name='CatBreed')
        ]).to(device)


    def _load_model(self, model_class, weights_path, *args, **kwargs):
        model = model_class(*args, **kwargs).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()
        return model

    def get_breed(self, image_tensor, animal_type):
        if animal_type == 1:
            pred = self.dog_model.predict(image_tensor)
            return pred.item()
        elif animal_type == 0:
            pred = self.cat_model.predict(image_tensor)
            return pred.item() + 25
        else:
            raise ValueError("animal_type должен быть 'dog' или 'cat'")

class SiameseNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 512
        self.breed_fc = nn.Linear(37, 64)
        self.merge = nn.Sequential(
            nn.Linear(self.feature_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def encode(self, img, breed):
        x = self.backbone(img)
        x = x.squeeze(-1).squeeze(-1)
        b = F.relu(self.breed_fc(breed))
        x = torch.cat([x, b], dim=1)
        return self.merge(x)

    def forward(self, img1, breed1, img2, breed2):
        emb1 = self.encode(img1, breed1)
        emb2 = self.encode(img2, breed2)
        return emb1, emb2

    def get_distance(self, tensor1, tensor2,
                     mode='euclidian', catdog_checker=None,
                     breed_extractor=None):
        if catdog_checker is None or breed_extractor is None:
            raise ValueError(
                "Необходимо передать catdog_checker, breed_extractor")

        self.eval()
        with torch.no_grad():

            is_dog1 = catdog_checker.predict(tensor1)
            # print(is_dog1)
            breed1_idx = breed_extractor.get_breed(tensor1, is_dog1)


            is_dog2 = catdog_checker.predict(tensor2)

            # print(is_dog2)
            breed2_idx = breed_extractor.get_breed(tensor2, is_dog2)

            breed1 = F.one_hot(torch.tensor(breed1_idx), num_classes=37).float().unsqueeze(0).to(device)
            breed2 = F.one_hot(torch.tensor(breed2_idx), num_classes=37).float().unsqueeze(0).to(device)

            out1, out2 = self.forward(tensor1, breed1, tensor2, breed2)
            # print(out1, out2)
            if mode == 'euclidian':
                distance = F.pairwise_distance(out1, out2).item()
            else:
                distance = (1 - F.cosine_similarity(out1, out2)).item()

            return distance

PetsVsOther18 = ModelWrapper(ResNet18, './checker/catsdogs_vs_others_ResNet18.pth', num_classes=2, name='Resnet18-1')
PetsVsOther34 = ModelWrapper(ResNet34, './checker/catsdogs_vs_others_ResNet34.pth',  num_classes=2, name='Resnet34-1')
Recognizer = EnsembleModel([PetsVsOther18, PetsVsOther34]).to(device)

CatsVsDogs18 = ModelWrapper(ResNet18, './checker/cats_vs_dogs_ResNet18.pth', num_classes=2, name='Resnet18-2')
CatsVsDogs34 = ModelWrapper(ResNet34, './checker/cats_vs_dogs_ResNet34.pth', num_classes=2, name='Resnet34-2')
CatsVsDogs = EnsembleModel([CatsVsDogs18, CatsVsDogs34]).to(device)

BreedExtractor = BreedExtractor(
    dog_weights_paths=["./feature_extractors/dogs_breed18.pth", "./feature_extractors/dogs_breed34.pth"],
    cat_weights_paths=["./feature_extractors/cats_breed18.pth", "./feature_extractors/cats_breed34.pth"]
)

Identifier = SiameseNet('Siamese').to(device)
Identifier.load_state_dict(torch.load('./identifier/siamese_cosine.pth', weights_only=True))