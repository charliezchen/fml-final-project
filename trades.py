import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from IPython import embed as e

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                model_ref,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                lam=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    adv_logits = model(x_adv)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                    F.softmax(logits, dim=1))
    if model_ref:
        ref_logits = model_ref(x_adv)
        # from IPython import embed as e
        # e() or b
        mask=1-F.one_hot(y, 10)
        # logits[torch.arange(logits.shape[0]), y] = 0
        # ref_logits[torch.arange(ref_logits.shape[0]), y] = 0
        # separation_loss = F.mse_loss(adv_logits.view(-1), ref_logits.view(-1))
        separation_loss = (F.softmax(adv_logits- ref_logits)*mask).square().mean()
        # print("separation loss", separation_loss)
    else:
        separation_loss = 0

    loss = loss_natural + beta * loss_robust - lam * separation_loss
    return loss


def sep_loss(models,
            model_ref_index,
            x_natural,
            y,
            optimizers,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            beta=1.0,
            lam=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)

    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    
    # Generate adversarial example
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(models[model_ref_index](x_adv), dim=1),
                                    F.softmax(models[model_ref_index](x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)


    models[model_ref_index]
    l_tilde_ref = models[model_ref_index](x_adv)
    l_tilde_ref_no_grad = l_tilde_ref.clone().detach()

    losses = []
    for i in range(len(models)):
        loss = 0
        l = models[i](x_natural)
        loss_natural = F.cross_entropy(l, y)
        loss += loss_natural

        if i == model_ref_index:
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(l_tilde_ref, dim=1),
                                                    F.softmax(l, dim=1))
            loss += beta * loss_robust
        else:
            l_tilde = models[i](x_adv)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(l_tilde, dim=1),
                                                    F.softmax(l, dim=1))
            
            mask=1-F.one_hot(y, 10)
            loss_sep = F.cosine_similarity(l_tilde * mask, l_tilde_ref_no_grad * mask).mean()
            loss += lam * loss_sep
        
        losses.append(loss)
    # e() or b
    return losses