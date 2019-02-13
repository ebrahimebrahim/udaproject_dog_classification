from PIL import Image
from torchvision import transforms
import torch
from torch import nn
import torchvision.models as models

# Check if cuda is available
use_cuda = torch.cuda.is_available()

# Set up transforamtions
normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean=-np.array(normalize.mean) * (1/np.array(normalize.std)),
                                   std=1/np.array(normalize.std))
transformation = transforms.Compose([transforms.Resize((224,224)),
                                             transforms.ToTensor(),
                                             normalize,])

# Load up human_dog_net
human_dog_net = models.vgg16(pretrained=False)
human_dog_net.classifier = nn.Sequential(nn.Linear(25088,4096),
                                          nn.ReLU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(4096,512),
                                          nn.ReLU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(512,3),
                                          nn.LogSoftmax(dim=1),
                                         )
if use_cuda:
    human_dog_net = human_dog_net.cuda()
human_dog_net.load_state_dict(torch.load('neural_networks/human_dog_net.pt', map_location='cuda' if use_cuda else 'cpu'))

# Load up model_transfer
model_transfer = models.vgg16(pretrained=False)
model_transfer.classifier[3]=nn.Linear(4096,512).to('cuda' if use_cuda else 'cpu')
model_transfer.classifier[6]=nn.Linear(512,133).to('cuda' if use_cuda else 'cpu')
model_transfer.classifier[2]=nn.BatchNorm1d(4096).to('cuda' if use_cuda else 'cpu')
model_transfer.classifier[5]=nn.BatchNorm1d(512 ).to('cuda' if use_cuda else 'cpu')
model_transfer.classifier.add_module("7",nn.LogSoftmax(dim=1))
model_transfer.load_state_dict(torch.load('neural_networks/model_transfer.pt'))

idx_to_class = pickle.load(open('idx_to_class.p','rb')) # this was generated in jupyter notebook

def dog_breed_net(f):
  str_out=""
  img = testing_transformation(Image.open(f).convert('RGB')).unsqueeze(dim=0)
  if use_cuda: img=img.cuda()
  human_dog_net.eval()
  model_transfer.eval()
  with torch.no_grad():
    human_dog_presence = human_dog_net(img).argmax().item()
  if human_dog_presence == 0:
    str_out+="I can't find the dog in this image.<br>"
    return str_out
  elif human_dog_presence == 1:
    str_out+="That's probably a ... "
  elif human_dog_presence == 2:
    str_out+="I can't find a dog, but there does seem to be a human in this image!<br>"
    str_out+="The image most closely resembles a ... "
  with torch.no_grad():
     pred=torch.exp(model_transfer(img))
  probabilities,classes=pred.squeeze().topk(5,sorted=True)
  classes=classes.cpu()
  class_names = [idx_to_class[idx] for idx in classes.data.numpy().tolist()]
  str_out+=class_names[0]+"."
  if probabilities[0] < 0.9:
    str_out+="<br><br> But I'm not very confident. Maybe it's a "
    str_out+=class_names[1]+"? Or a "+class_names[2]+"?"
  return str_out







