
import numpy as np
import torch_directml
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('../'))
from task_vectors import TaskVector
from evaluation import eval_single_dataset

# Caminhos dos checkpoints
pretrained_checkpoint = 'artificial_checkpoints/mlp_model.pth'
finetuned_checkpoint1 = 'artificial_checkpoints/mlp_model_task1_dataset.pth'
finetuned_checkpoint2 = 'artificial_checkpoints/mlp_model_task2_dataset.pth'
finetuned_checkpoint3 = 'artificial_checkpoints/mlp_model_task3_dataset.pth'
device = torch_directml.device(torch_directml.default_device())

# Criação dos task vectors (imutáveis)
task_vector1 = TaskVector(pretrained_checkpoint=pretrained_checkpoint, finetuned_checkpoint=finetuned_checkpoint1, operation='subtract')
task_vector2 = TaskVector(pretrained_checkpoint=pretrained_checkpoint, finetuned_checkpoint=finetuned_checkpoint2, operation='subtract')
task_vector3 = TaskVector(pretrained_checkpoint=pretrained_checkpoint, finetuned_checkpoint=finetuned_checkpoint3, operation='subtract')

def new_loss(b1, b2, b3):
    # Combinação ponderada dos task vectors
    combined_vector = TaskVector(vector={})
    for key in task_vector1.vector:
        v1 = task_vector1.vector.get(key, 0)
        v2 = task_vector2.vector.get(key, 0)
        v3 = task_vector3.vector.get(key, 0)
        combined_vector.vector[key] = b1 * v1 + b2 * v2 + b3 * v3

    # Aplica o vetor combinado ao modelo base
    model_backbone = combined_vector.apply_to(pretrained_checkpoint, 1.0)

    # Avalia nos quatro datasets
    datasets = ['task1_dataset', 'task2_dataset', 'task3_dataset']
    accs = []
    for ds in datasets:
        acc = eval_single_dataset(model_backbone, ds, device)
        accs.append(acc)

    # Retorna a média das acurácias
    return np.mean(accs)

# Inicializa variáveis de controle
b1 = 1/3
b2 = 1/3
b3 = 1/3
T = 100
i = 0
dmax = 0.1
loss_candidata = 0


# Vetor de otimização
for i in range(100):
    b1_cand = b1
    b2_cand = b2
    b3_cand = b3
    d = np.random.uniform(-1, 1)*dmax

    if i%3 == 0: #d12
        b1_cand = b1 + d
        b2_cand = b2 - d
    elif i%3 == 1: #d13
        b1_cand = b1 + d
        b3_cand = b3 - d
    else: #d23
        b2_cand = b2 + d
        b3_cand = b3 - d
    new_l = new_loss(b1_cand, b2_cand, b3_cand)
    if (new_l > loss_candidata):
        if b1_cand < 0 or b2_cand < 0 or b3_cand < 0:
            print(f"Invalid candidate: b1: {b1_cand}, b2: {b2_cand}, b3: {b3_cand}. Skipping.")
            continue

        b1 = b1_cand
        b2 = b2_cand
        b3 = b3_cand
        loss_candidata = new_l
        print(f"New best loss: {loss_candidata} with b1: {b1}, b2: {b2}, b3: {b3}")
    else:
        prob = np.exp((new_l - loss_candidata) / T)
        if np.random.rand() < prob:
            if b1_cand < 0 or b2_cand < 0 or b3_cand < 0:
                print(f"Invalid candidate: b1: {b1_cand}, b2: {b2_cand}, b3: {b3_cand}. Skipping worst loss.")
                continue
            b1 = b1_cand
            b2 = b2_cand
            b3 = b3_cand
            loss_candidata = new_l
            print(f"Accepted worse loss: {loss_candidata} with b1: {b1}, b2: {b2}, b3: {b3} at i {i} epoch and T: {T}")

    T = T * 0.99  # Decaimento da temperatura
    dmax = dmax * 0.99  # Decaimento do passo





