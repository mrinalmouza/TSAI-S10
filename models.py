import torch.nn as nn
import torch.nn.functional as F
import pdb


class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()

        #Input image = [B, 3, 32, 32]
        self.prep_layer     = nn.Sequential(
                                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding= 1, stride=1, bias = False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        #nn.Dropout(.02),
                                        )
        #Output of prep layer = [B, 64, 32, 32], Receptive Field = 3
        self.layer1         = nn.Sequential(
                                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, bias = False),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        # nn.Dropout(.04),    
                                        )
        #Output of layer1 = [B, 128, 16, 16], Receptive field = 6
        self.residual_block1 = nn.Sequential(
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,  padding =1, stride=1, bias = False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        #nn.Dropout(.01),
                                        nn.Conv2d(in_channels=128, out_channels= 128, kernel_size = 3, padding=1, stride=1, bias = False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        # nn.Dropout(.01),    
                                        )
        #Output of layer1 = [B, 128, 16, 16], Receptive field = 14
        self.layer2         = nn.Sequential(
                                        #nn.ReLU(),
                                        nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, padding=1, stride=1, bias = False),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # nn.Dropout(.15),

                                        )
        #Output of layer1 = [B, 256, 8, 8], Receptive field = 18
        self.layer3         = nn.Sequential(
                                        nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3, stride=1, padding=1, bias = False),
                                        nn.MaxPool2d(2,2),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        # nn.Dropout(.20),
                                        )     
        #Output of layer1 = [B, 512, 4, 4], Receptive field = 26
        self.residual_block2 = nn.Sequential(
                                        nn.Conv2d(in_channels=512, out_channels=512,kernel_size=3, padding =1, stride=1, bias = False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                       # nn.Dropout(.01),
                                        nn.Conv2d(in_channels=512, out_channels= 512,kernel_size=3, padding=1, stride=1, bias = False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        # nn.Dropout(.01),    
                                        )
        #Output of layer1 = [B, 512,4, 4], Receptive field = 32
        #self.non_linear = nn.ReLU()

        self.maxpool = nn.MaxPool2d(4, 4)

        #Output of layer1 = [B, 512, 1, 1]

        # self.layer4 = nn.Sequential(nn.Linear(8192, 512),
        #                             nn.BatchNorm1d(512),
        #                             nn.ReLU(),
        #                             nn.Dropout(.02),
        #                             nn.Linear(512, 50),
        #                             nn.BatchNorm1d(50),
        #                             nn.ReLU(),
        #                             nn.Dropout(.02),
        # )
        self.fc1 = nn.Linear(512, 10, bias = False)



    def forward(self, x):
        #pdb.set_trace()
        x = self.prep_layer(x)
        x = self.layer1(x)
        #pdb.set_trace()
        r1 = self.residual_block1(x)
        x1 = x + r1
        # x = F.relu(x1)
        x = self.layer2(x1)
        x = self.layer3(x)
        r2 = self.residual_block2(x)
        x2 = x+ r2
        # x = F.relu(x2)
        #x2 = self.non_linear(x2)
        x = self.maxpool(x2)
        x = x.view(-1, 512)

        #x = self.layer4(x)
        
        x = self.fc1(x)
        
        #x = F.log_softmax(x, dim=1)

        return x