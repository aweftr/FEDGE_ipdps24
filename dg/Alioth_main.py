# %%
import os
import argparse
import logging
import pickle
import torch
# set torch random seed so that the result is reproducible
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from utils import *
from qos_datasets import *
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# %%
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=1) -> None:
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.fc_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim=1) -> None:
        super().__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.fc_stack(x)
        return x

class DAE(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.encoder = Encoder(dim, 128)
        self.decoder = Decoder(128, dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''
    def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.domain_classifier = Discriminator(**kwargs)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        
    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss
    
    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv
    

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class ReconstructionLoss(nn.Module):
    def __init__(self, decoder_sizes, output_size) -> None:
        super(ReconstructionLoss,self).__init__()
        self.decoder_sizes = decoder_sizes
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_sizes) - 1):
            self.decoder.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(decoder_sizes[-1], output_size))
        self.decoder.append(nn.ReLU())

    def forward(self, x, y):
        for layer in self.decoder:
            x = layer(x)
        loss = F.mse_loss(x, y)
        return loss
    
    def load_state_dict_from_DAE(self, filename):
        '''
        Partially load the state dictionary from a DAE, only using params of its deocder
        '''
        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'decoder' in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Loaded pretrained decoder from {}'.format(filename))

# %%
# setting up logger
logfile = "./log/Alioth.log"
make_file_dir(logfile)
logger = logging.getLogger("Alioth logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Alioth for prediction")
    parser.add_argument('--target', type=int, default=7, metavar='N',
                        help='target domain number. 0 to 7.')
    parser.add_argument("--output", type=str,
                        default='output/Alioth_output.csv', help="output file")
    parser.add_argument("--feature", type=str, default="all", help="feature file")
    parser.add_argument('--save_finalresult', type=int, default=1, metavar='N',
                        help='save final result or not')
    # args = parser.parse_args([])
    args = parser.parse_args()
    return args


args = get_parameters()
logger.info(args)
target_app = args.target
features = args.feature
batch_size = 100
test_batch_size = 4196
train_target_batch_size = 700
lr = 0.001
iterations = 2000
data_dir = "data/dg-data"
app_list = ["cassandra", "etcd", "hbase", "kafka", "milc", "mongoDB", "rabbitmq", "redis"]
model_save_path = "model/Alioth"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# generate remote DG dataset for torch
train_loaders = []
val_loaders = []
for idx, app in enumerate(app_list):
    with open(os.path.join(data_dir, "{}.pickle".format(app)), "rb") as f:
        full, train, val = pickle.load(f)
    if idx == target_app:
        test_loader = DataLoader(full, batch_size=test_batch_size)
        train_target_loader = DataLoader(full, batch_size=train_target_batch_size, shuffle=True)
        continue
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=test_batch_size)
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)

# %%
def save_finalresult(args, test_loss):
    make_file_dir(args.output)
    with open(args.output, mode="a+") as f:
        f.write(str(args.target) + ", ")
        f.write(str(test_loss))
        f.write("\n")

def pretrain_DAE():
    encoder_model.train()
    decoder_model.train()
    adaptation_criterion.train()
    dae_ites = 200
    alpha = 0.5
    logger.info("DAE start pretraining.")
    for ite in range(dae_ites):
        datas = []
        labels = []
        for i in range(len(train_loaders)):
            app, nostress, noapp, label = next(iter(train_loaders[i]))
            datas.append(app)
            labels.append(nostress)
        target = next(iter(train_target_loader))[0].to(device)
        datas = torch.cat(datas, dim=0).to(device)
        labels = torch.cat(labels, dim=0).to(device)
        optimizer.zero_grad()

        enc_outputs = encoder_model(datas)
        rec_outputs = decoder_model(enc_outputs)
        target_outputs = encoder_model(target)

        # print(rec_outputs.shape, labels.shape, enc_outputs.shape, target_outputs.shape)
        loss = reconstruction_criterion(rec_outputs, labels) + alpha * adaptation_criterion(enc_outputs, target_outputs)
        loss.backward()
        optimizer.step()
        logger.info("Iteration {}, train loss {:.6f}".format(ite, loss.item()))

def DAE_get_parameters(initial_lr):
    return [
        {"params": encoder_model.parameters(), "lr": 0.2 * initial_lr},
        {"params": decoder_model.parameters(), "lr": 1.0 * initial_lr},
        {'params': adaptation_criterion.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
    ]

# %%
app_dim = next(iter(test_loader))[0].shape[1]
noapp_dim = next(iter(test_loader))[2].shape[1]

encoder_model = Encoder(app_dim, 128).to(device)
decoder_model = Decoder(128, app_dim).to(device)
adaptation_criterion = AdversarialLoss().to(device)

reconstruction_criterion = nn.MSELoss()
optimizer = optim.Adam(DAE_get_parameters(lr), lr=lr)


# %%
pretrain_DAE()
# %%
def DAE_trans(app_data, noapp_data):
    encoder_model.eval()
    decoder_model.eval()
    data = torch.Tensor(app_data).to(device)
    dae_output = decoder_model(encoder_model(data)).detach().cpu().numpy()
    return np.concatenate([app_data, dae_output, noapp_data], axis=1)

train_datas = []
train_labels = []
val_datas = []
val_labels = []
for idx, app in enumerate(app_list):
    with open(os.path.join(data_dir, "{}.pickle".format(app)), "rb") as f:
        full, train, val = pickle.load(f)
    if idx == target_app:
        app_test, noapp_test, y_test = alioth_data(full)
        x_test = DAE_trans(app_test, noapp_test)
        continue
    app_train, noapp_train, y_train = alioth_data(train)
    app_val, noapp_val, y_val = alioth_data(val)
    x_train = DAE_trans(app_train, noapp_train)
    x_val = DAE_trans(app_val, noapp_val)
    train_datas.append(x_train)
    train_labels.append(y_train)
    val_datas.append(x_val)
    val_labels.append(y_val)
    logger.info("app: {}, train shape: {}".format(app, x_train.shape))

x_train = np.concatenate(train_datas, axis=0)
y_train = np.concatenate(train_labels, axis=0)
x_val = np.concatenate(val_datas, axis=0)
y_val = np.concatenate(val_labels, axis=0)

logger.info("Total train shape: {}".format(x_train.shape))
logger.info("Total val shape: {}".format(x_val.shape))
logger.info("Total test shape: {}".format(x_test.shape))

# %%
model = XGBRegressor()
reg = model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_test_pred = model.predict(x_test)

train_loss_mse = mean_squared_error(y_train, y_train_pred)
val_loss_mse = mean_squared_error(y_val, y_val_pred)
test_loss_mse = mean_squared_error(y_test, y_test_pred)

train_loss_mae = mean_absolute_error(y_train, y_train_pred)
val_loss_mae = mean_absolute_error(y_val, y_val_pred)
test_loss_mae = mean_absolute_error(y_test, y_test_pred)

logger.info("Train mse loss: {:.4e}, Val mse loss: {:.4e}, Test mse loss: {:.4e}".format(train_loss_mse, val_loss_mse, test_loss_mse))
logger.info("Train mae loss: {:.4e}, Val mae loss: {:.4e}, Test mae loss: {:.4e}".format(train_loss_mae, val_loss_mae, test_loss_mae))
if args.save_finalresult:
    save_finalresult(args, test_loss_mae)