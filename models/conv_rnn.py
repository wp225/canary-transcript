from torch import nn

class ConvRNNSegmentor(nn.Module):
    def __init__(self,
                 chan1=32,
                 chan2=64,
                 chan3=64,
                 k_size=11,
                 bi=True,
                 emb_size=128,
                 n_rnn_layers=1,
                 p_dropout=0.4,
                 ):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, chan1, 11, padding="same", bias=True), nn.LeakyReLU(), nn.MaxPool2d((8, 1), (8, 1)), nn.Dropout2d(p_dropout),
                                 nn.Conv2d(chan1, chan2, 5, padding="same", bias=True), nn.LeakyReLU(), nn.MaxPool2d((4, 1), (4, 1)), nn.Dropout2d(p_dropout),
                                 nn.Conv2d(chan2, chan3, 3, padding="same", bias=True), nn.LeakyReLU(), nn.MaxPool2d((2, 1), (2, 1)), nn.Dropout2d(p_dropout/2))
        self.rnn = nn.LSTM(256, emb_size, n_rnn_layers, batch_first=True, bidirectional=bi, dropout=0.25 if n_rnn_layers > 1 else 0)
        self.fc = nn.Linear(emb_size * 2 if bi else emb_size, 1)  # * 2 since bidirectional doubles the output size of rnn
        self.fc_drop = nn.Dropout1d(p_dropout)

    def forward(self, x):
        self.rnn.flatten_parameters()
        batch_size, channels, freq_bins, time_bins = x.shape
        x = self.cnn(x)
        x = x.contiguous().view(batch_size, 256, time_bins).transpose(1, 2).contiguous()  # add .contiguous() here
        x, *_ = self.rnn(x)
        x = self.fc_drop(x.transpose(1, 2)).transpose(1, 2)
        x = self.fc(x)
        return x
    
    def extract_features(self, x):
        self.rnn.flatten_parameters()
        batch_size, channels, freq_bins, time_bins = x.shape
        x = self.cnn(x)                                 
        x = x.view(batch_size, 256, time_bins).transpose(1, 2)  
        x, *_ = self.rnn(x)                              
        x = x.mean(dim=1)                                  # (B, 256)
        return x