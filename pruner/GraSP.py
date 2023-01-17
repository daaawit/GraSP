import copy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y

def GraSP(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []

    # rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)
        
    for it in range(num_iters):
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        
        # I don't fully understand why this is happening
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = net.forward(inputs[:N//2])/T
        loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:])/T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []

    for it in range(len(inputs_one)):
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg
    
    ###
    # From here on out, the code was essentially identical to mil-ad/snip, but 
    # more complicated, using dicts instead of lists. I modified this, but it 
    # might affect training (which I didn't check for)
    ###
     
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps # TODO: *What* is this? Absolute sum? Why? Where is this in the paper? 
    all_scores.div_(norm_factor)

    num_params_to_keep = int(len(all_scores) * (keep_ratio))
    
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    
    keep_masks = list()
    for _, g in grads.items():
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())
    
    names = [name for name, _ in net.named_modules()]
    numel, numprune = print_scores(keep_masks, names)
    
    print(f"- Intended prune ratio:\t{1-keep_ratio}")
    print(f"- Actual prune ratio:\t{1 - (numprune / numel)}")
    print(f"- Threshold:           {acceptable_score}")

    return keep_masks

def print_scores(keep_masks, names):
    """Printing in style of my master's thesis for sanity checking

    Args:
        keep_masks (list): List of masks produced by SNIP
        names (_type_): List of names of pruned layers (for printing style only)
    """
    head_str = f"| {'Layer':<17}| {'Before':<14}| {'After':<14}| {'Ratio':<10}|"
    head_sep = "=" * len(head_str)
    print(head_sep)
    print(head_str)
    print(head_sep)
    
    full_numel = 0
    full_numprune = 0
    for name, mask in zip(names, keep_masks): 
        numel = torch.numel(mask)
        numprune = torch.sum(mask).numpy()
        ratio = str(np.round(numprune / numel, 4))
        
        layer_info = f"| - {name:<15}| {numel:<14}| {numprune:<14}| {ratio:<10}|"
        print(layer_info)
        
        full_numel += numel
        full_numprune += numprune
    
    print(head_sep, "\n")
    return full_numel, full_numprune