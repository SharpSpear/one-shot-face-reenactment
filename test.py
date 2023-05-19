import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from PIL import Image
import torch
print('cuda available??',torch.cuda.is_available())
#from torchsummary import summary
from models.psp import pSp
import numpy as np
from configs.transforms_config import EncodeTransforms
from options.train_options import TrainOptions
opts = TrainOptions().parse()
opts.device = 'cpu'
def visual(output):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    plt.figure(1)
    plt.axis('off')
    plt.imshow(output)
    plt.show()
    # plt.savefig('reconstrucetd.png')
net = pSp(opts)
net.eval()
net.cuda()
print('net',net)
x = Image.open('image_1.png').convert('RGB')
trans = EncodeTransforms(opts).get_transforms()['transform_test']
x = trans(x)
x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2]).cuda().float()
# x = torch.randn(1,3,256,256).cuda()
images, latent = net.forward(x, resize=False, return_latents=True)
visual(images)
