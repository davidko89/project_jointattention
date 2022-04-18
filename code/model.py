'''
CNN_LSTM binary classification (TD vs ASD) 
Transfer learning using VGG16 in Pytorch
'''
import torch
import torch.nn as nn
import torchvision.models as models

class VideoRNN(nn.Module):
    def __init__(self, n_classes, batch_size, device):
        super(VideoRNN, self).__init__()
        self.batch_size = batch_size
        self.device = device
        
        # Loading vgg16
        vgg = models.vgg16(pretrained=True)
        
        # Removing last layer of vgg16
        embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.classifier = embed

        # Freezing the model last 3 layers
        for param in vgg.parameters():
            param.requires_grad = False

        self.embedding = vgg
        self.gru = nn.LSTM(4096, 2048, bidirectional = True)

        # Classification layer (*2 because bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2 , 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, input):
            hidden = torch.zeros(2, self.batch_size, 2048).to(self.device)

            c_0 = torch.zeros(self.num_layer * 2, self.batch_size, 2048).to(self.device)

            embedded = self.simple_elementwise_apply(self.embedding, input)
            output, hidden = self.gru(embedded, (hidden, c_0))
            hidden = hidden[0].view(-1, 2048 * 2)

            output = self.classifier(hidden)

            return output

    def simple_elementwise_apply(self, fn, packed_sequence):
        return torch.nn.utils.rnn.PackedSequence(
            fn(packed_sequence.data), packed_sequence.batch_sizes
        )

if __name__ =='__main__':
    VideoRNN()