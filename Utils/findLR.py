import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist

def find_lr(model, optimizer, start_val = 1e-6, end_val = 1, beta = 0.99, loader = DataLoader, rank=0, world_size=1):
    n = len(loader) - 1
    factor = (end_val / start_val)**(1/n)
    lr = start_val
    optimizer.param_groups[0]['lr'] = lr  # Esto permite actualizar la tasa de aprendizaje
    avg_loss, loss, acc = 0., 0., 0.
    lowest_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []
    accuracies = []
    
    # Asegúrate de usar la GPU correspondiente al rank
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        scores = model(x)
        
        cost = F.cross_entropy(input=scores, target=y)
        loss = beta * loss + (1 - beta) * cost.item()
        
        # Corrección de sesgo
        avg_loss = loss / (1 - beta**i)
        
        acc_ = ((torch.argmax(scores, dim=1) == y).sum() / scores.size(0))
        
        # Si la pérdida se vuelve masiva, detener la búsqueda
        if i > 1 and avg_loss > 4 * lowest_loss:
            print(f"De aquí en adelante, la pérdida es demasiado alta: {i}, Pérdida: {cost.item()}")
            return log_lrs, losses, accuracies
        
        if avg_loss < lowest_loss or i == 1:
            lowest_loss = avg_loss
        
        accuracies.append(acc_.item())
        losses.append(avg_loss)
        log_lrs.append(lr)
        
        # Retropropagar y actualizar los pesos
        cost.backward()
        optimizer.step()
        
        # Actualizar la tasa de aprendizaje
        lr *= factor
        optimizer.param_groups[0]['lr'] = lr
        
    # Sincronizar las métricas de pérdida y precisión entre los diferentes procesos DDP
    if world_size > 1:
        # Reducir la pérdida y precisión en todos los procesos
        loss_tensor = torch.tensor(losses[-1]).to(device)
        acc_tensor = torch.tensor(accuracies[-1]).to(device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        
        # Promediar las métricas a través de todos los procesos
        loss_tensor /= world_size
        acc_tensor /= world_size
        
        # Actualizar las métricas globales
        losses[-1] = loss_tensor.item()
        accuracies[-1] = acc_tensor.item()
    
    return log_lrs, losses, accuracies
 