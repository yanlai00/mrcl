import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner
from copy import deepcopy

class MemoryMetaLearner(nn.Module):

    def __init__(self, model_config, adapt_lr, meta_lr):
        super(MemoryMetaLearner, self).__init__()

        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr

        self.net = Learner.Learner(model_config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def reset_classifer(self, class_to_reset):
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self):
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight)

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates
        x_traj, y_traj, x_rand, y_rand, x_rand_temp, y_rand_temp = [], [], [], [], [], []
        counter = 0
        class_counter = 0
        for it1 in iterators:
            steps_inner = 0
            rand_counter = 0
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    self.reset_classifer(class_to_reset)
                counter += 1
                batch_size = img.shape[0]
                img_flatten = img.view(batch_size, -1)
                one_hot_label = F.one_hot(data, num_classes=1000)
                concat_img_label = torch.cat((img_flatten, one_hot_label), dim=1)
                if steps_inner < steps:
                    x_traj.append(concat_img_label)
                    y_traj.append(data)
                    steps_inner += 1
                else:
                    x_rand_temp.append(concat_img_label)
                    y_rand_temp.append(data)
                    rand_counter += 1
                    if rand_counter == 5:
                        break
            class_counter += 1

        # Sampling the random batch of data
        for img, data in it2:
            
            # import pdb; pdb.set_trace()
            batch_size = img.shape[0]

            img_flatten = img.view(batch_size, -1)
            one_hot_label = F.one_hot(data, num_classes=1000)
            concat_img_label = torch.cat((img_flatten, one_hot_label), dim=1)

            x_rand.append(concat_img_label)
            y_rand.append(data)
            break

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)

        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        return x_traj, y_traj, x_rand, y_rand

    def inner_update(self, x, fast_weights, y):
        y = deepcopy(x)
        adaptation_weight_counter = 0

        logits = self.net(x, fast_weights)
        loss = F.mse_loss(logits, y)
        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = torch.autograd.grad(loss, self.net.get_adaptation_parameters(fast_weights), create_graph=False)
        grad = self.clip_grad(grad)

        # Gradient descent on the adaptation layers
        new_weights = []
        for p in fast_weights:
            if p.adaptation:
                g = grad[adaptation_weight_counter]
                temp_weight = p - self.adapt_lr * g
                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta
                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

    def meta_loss(self, x, fast_weights, y):
        logits = self.net(x, fast_weights)[:, 1024:]
        loss_q = F.cross_entropy(logits, y)
        return loss_q

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        # Doing a single inner update to get updated weights
        fast_weights = self.inner_update(x_traj[0], None, y_traj[0])

        for k in range(1, len(x_traj)):
            # Doing inner updates using fast weights
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k])

            # Computing meta-loss with respect to latest weights
            meta_loss = self.meta_loss(x_rand[0], fast_weights, y_rand[0])

        # Taking the meta gradient step
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        return meta_loss

    def clip_grad(self, grad, norm=10):
        grad_clipped = []
        for g, p in zip(grad, self.net.parameters()):
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            grad_clipped.append(g)
        return grad_clipped
