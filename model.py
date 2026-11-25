import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def load_model(device):
    weights = ResNet18_Weights.DEFAULT
    net = resnet18(weights=weights)
    return torch.nn.Sequential(*list(net.children())[:-2]).eval().to(device), weights.transforms()

@torch.inference_mode()
def features(model, images):
    x = model(images).flatten(2)
    g = F.normalize(x.mean(2), dim=1)
    # Activation magnitude is a spatial saliency proxy, not a semantic part detector.
    w = torch.softmax(x.norm(dim=1) / (x.norm(dim=1).mean(1, keepdim=True)+1e-8), dim=1)
    l = F.normalize((x*w[:,None,:]).sum(2), dim=1)
    return g, l

def scores(support, query, metric='cosine'):
    p = support.mean(1)
    if metric == 'euclidean':
        return -torch.cdist(query,p).square()
    return F.normalize(query,dim=1) @ F.normalize(p,dim=1).T
