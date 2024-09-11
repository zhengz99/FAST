import torch
import torch.nn.functional as F
import random
from torch.func import functional_call
from torch import nn


class item(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_autor = 28678
        self.num_publisher = 5897
        self.num_year = 1170

        self.embedding_size = args.embedding_size

        self.author_embedding = torch.nn.Embedding(
            self.num_autor,
            self.embedding_size
        )

        self.publisher_embedding = torch.nn.Embedding(
            self.num_publisher,
            self.embedding_size
        )

        self.year_embedding = torch.nn.Embedding(
            self.num_year,
            self.embedding_size
        )
    
    def forward(self, author_idx, publisher_idx, year_idx):
        author_emb = self.author_embedding(author_idx)
        publisher_emb = self.publisher_embedding(publisher_idx)
        year_emb = self.year_embedding(year_idx)

        return torch.cat((author_emb, publisher_emb, year_emb), dim=1)
    
class user(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_location = 58
        self.num_age = 2

        self.embedding_size = args.embedding_size

        self.location_embedding = torch.nn.Embedding(
            self.num_location,
            self.embedding_size
        )

        self.age_embedding = torch.nn.Embedding(
            self.num_age,
            self.embedding_size
        )
    
    def forward(self, location_idx, age_idx):
        location_emb = self.location_embedding(location_idx)
        age_emb = self.age_embedding(age_idx)

        return torch.cat((location_emb, age_emb), dim=1)


class user_preference_estimator(torch.nn.Module):

    def __init__(self, args):

        super().__init__()
        self.item = item(args)
        self.user = user(args)

        embedding_sz = args.embedding_size

        fc1_input_size = 64
        fc2_input_size = 64
        fc2_output_size = 64

        self.item_fc = torch.nn.Linear(3 * args.embedding_size, fc1_input_size // 2)
        self.user_fc = torch.nn.Linear(2 * args.embedding_size, fc1_input_size // 2)

        self.fc1 = torch.nn.Linear(fc1_input_size, fc2_input_size)
        self.fc2 = torch.nn.Linear(fc2_input_size, fc2_output_size)
        self.linear_output = torch.nn.Linear(fc2_output_size, 1)

    def forward(self, x):

        author_idx = x[:,0]
        publisher_idx = x[:,1]
        year_idx = x[:,2]
        location_idx = x[:,3]
        age_idx = x[:,4]

        item_emb = self.item(author_idx, publisher_idx, year_idx)
        user_emb = self.user(location_idx, age_idx)

        item_emb = F.relu(self.item_fc(item_emb))
        user_emb = F.relu(self.user_fc(user_emb))

        x = torch.cat((item_emb, user_emb), dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.linear_output(x)

class MetaLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_updated_parameters = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weigh', 'linear_out.bias', 'item_fc.weight', 'item_fc.bias', 'user_fc.weight', 'user_fc.bias']
        self.local_lr = torch.tensor(config.local_lr, requires_grad=True)
        self.device = config.device

        self.model = user_preference_estimator(config).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.optim_innerLR = torch.optim.Adam([self.local_lr], lr=config.lr)

    def forward(self, supp_x, supp_y, qry_x):

        fast_parameter = {}

        y_hat = functional_call(self.model, dict(self.model.named_parameters()), supp_x)
        loss = F.mse_loss(y_hat, supp_y)
        grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        for g, (n, p) in zip(grad, self.model.named_parameters()):
            if n in self.local_updated_parameters:
                fast_parameter[n] = p - self.local_lr * g
            else:
                fast_parameter[n] = p
        
        qry_y_hat = functional_call(self.model, fast_parameter, qry_x)

        return qry_y_hat

    def global_update(self, batch_xs, batch_ys):

        batch_sz = len(batch_xs)

        qry_losses = []
        for idx in range(batch_sz):

            num_records = len(batch_xs[idx])
            indice = list(range(num_records))
            random.shuffle(indice)

            supp_x = batch_xs[idx][indice[:-10]].to(self.device)
            supp_y = batch_ys[idx][indice[:-10]].to(self.device)
            qry_x = batch_xs[idx][indice[-10:]].to(self.device)
            qry_y = batch_ys[idx][indice[-10:]].to(self.device)


            qry_y_hat = self.forward(supp_x, supp_y, qry_x)
            qry_loss = F.mse_loss(qry_y_hat, qry_y)
            qry_losses.append(qry_loss)
        
        
        l = torch.stack(qry_losses).mean()

        self.optim.zero_grad()
        l.backward()
        self.optim.step()

        return l

    
# ==================================metaCS
    def global_update_metaCS(self, batch_xs, batch_ys):

        batch_sz = len(batch_xs)

        qry_losses = []
        for idx in range(batch_sz):

            num_records = len(batch_xs[idx])
            indice = list(range(num_records))
            random.shuffle(indice)

            supp_x = batch_xs[idx][indice[:-10]].to(self.device)
            supp_y = batch_ys[idx][indice[:-10]].to(self.device)
            qry_x = batch_xs[idx][indice[-10:]].to(self.device)
            qry_y = batch_ys[idx][indice[-10:]].to(self.device)


            qry_y_hat = self.forward(supp_x, supp_y, qry_x)
            qry_loss = F.mse_loss(qry_y_hat, qry_y)
            qry_losses.append(qry_loss)
        
        
        l = torch.stack(qry_losses).mean()

        self.optim.zero_grad()
        l.backward()
        self.optim.step()

        qry_losses = []
        for idx in range(batch_sz):

            num_records = len(batch_xs[idx])
            indice = list(range(num_records))
            random.shuffle(indice)

            supp_x = batch_xs[idx][indice[:-10]].to(self.device)
            supp_y = batch_ys[idx][indice[:-10]].to(self.device)
            qry_x = batch_xs[idx][indice[-10:]].to(self.device)
            qry_y = batch_ys[idx][indice[-10:]].to(self.device)


            qry_y_hat = self.forward(supp_x, supp_y, qry_x)
            qry_loss = F.mse_loss(qry_y_hat, qry_y)
            qry_losses.append(qry_loss)
        
        
        l = torch.stack(qry_losses).mean()

        self.optim_innerLR.zero_grad()
        l.backward()
        self.optim_innerLR.step()

        return l
