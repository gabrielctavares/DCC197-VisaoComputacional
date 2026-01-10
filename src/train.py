import os
import time
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from datetime import datetime
from sklearn.metrics import confusion_matrix

from log_util import save_results_to_excel, display_class_distribution, plot_confusion_matrix
from torchvision import datasets, transforms


def build_optimizer(model, lr, weight_decay, epochs):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    return optimizer, scheduler


def validate(model, dataloader, device, class_names):
    model.eval()
    correct, total = 0, 0
    correct_per_class = torch.zeros(len(class_names), dtype=torch.long, device=device)
    total_per_class   = torch.zeros(len(class_names), dtype=torch.long, device=device)
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total   += y.size(0)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for i in range(len(class_names)):
                mask = y == i
                total_per_class[i]   += mask.sum()
                correct_per_class[i] += (preds[mask] == i).sum()
            

    acc = correct / max(total, 1)
    class_accs = {
        i: (correct_per_class[i] / total_per_class[i]).item() if total_per_class[i] > 0 else 0.0
        for i in range(len(class_names))
    }    

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    return acc, class_accs, cm


def main(model_name, max_epochs, batch_size, img_size, learning_rate, weight_decay, use_dropout, dropout_rate, use_batch_norm, use_data_augmentation, freeze_features, unfreeze_last_n_layers, results_file="resultados.xlsx"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    num_workers=4
    pin_memory = is_cuda and (num_workers > 0)

    base_folder = os.path.dirname(os.path.abspath(__file__))
    output_model_path = os.path.join(base_folder, 'results')
    output_model_folder = os.path.join(output_model_path, f"{model_name}")
    os.makedirs(output_model_folder, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', handlers=[
        logging.FileHandler(os.path.join(output_model_folder, "train.log")),
        logging.StreamHandler()
    ])
    
    writer = SummaryWriter(log_dir=os.path.join(output_model_folder, "tensorboard"))

    logging.info(f"Starting training using {model_name} model, max epochs {max_epochs}, learning rate {learning_rate}, weight decay {weight_decay}, use dropout {use_dropout}, dropout rate {dropout_rate}, use batch norm {use_batch_norm}, use data augmentation {use_data_augmentation}, freeze features {freeze_features}, unfreeze last n layers {unfreeze_last_n_layers}.")

    transform_list = [transforms.Resize((img_size, img_size))]
    if use_data_augmentation:
        transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ]

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    transform_train = transforms.Compose(transform_list)
    

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )   

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    class_names = train_dataset.classes

    model = build_model(model_name, len(class_names), device=device, img_size=img_size, use_dropout=use_dropout, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, freeze_features=freeze_features, unfreeze_last_n_layers=unfreeze_last_n_layers).to(device)    

    display_class_distribution("Train", train_dataset, class_names)
    display_class_distribution("Test", test_dataset, class_names)

    optimizer, scheduler = build_optimizer(model, learning_rate, weight_decay, max_epochs)
    loss_fn = nn.CrossEntropyLoss().to(device)
   

    for epoch in range(max_epochs):        
        model.train()
        start_time = time.time()
        running_loss, running_acc, n_samples = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=is_cuda), y.to(device, non_blocking=is_cuda)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            bs = x.size(0)

            running_loss += loss.detach() * bs  
            preds = logits.argmax(dim=1)
            running_acc += (preds == y).sum()
            n_samples += bs

            avg_loss = (running_loss / n_samples).item()
            avg_acc  = (running_acc.float() / n_samples).item()
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc*100:.2f}%"})

        scheduler.step()
        train_loss = (running_loss / max(n_samples, 1)).item()
        train_acc  = (running_acc.float() / max(n_samples, 1)).item()        

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        logging.info(f"Epoch {epoch}: {time.time() - start_time:.2f}s")
        logging.info(f"  train loss:\t{train_loss:.4f}")
        logging.info(f"  train acc:\t{train_acc*100:.2f}%")        

    final_test_acc, test_class_accs, test_cm = validate(model, test_loader, device, class_names)
    writer.add_scalar("Accuracy/test", final_test_acc, max_epochs)
    logging.info(f"Final test acc:\t{final_test_acc*100:.2f}%")

    
    final_row = {
        "modelo": model_name,
        "test_acc": final_test_acc,
        **{f"test_{class_names[i]}": test_class_accs[i] for i in class_names},
        "batch_size": batch_size
    }   
    save_results_to_excel(results_file, final_row)

    torch.save({'model_state': model.state_dict()},
                os.path.join(output_model_folder, f"model_{batch_size}.pt"))  

    if test_cm is not None:
        fig = plot_confusion_matrix(test_cm, class_names, "Matriz de Confusão", os.path.join(output_model_folder, f"confusion_matrix_{batch_size}.pdf"))
        writer.add_figure("ConfusionMatrix/test", fig, max_epochs)        
    writer.close()
    
    logging.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="experiment_1")
    parser.add_argument("--model_name", type=str, default='VGG16')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    parser.add_argument("--use_dropout", type=bool, default=False)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--use_batch_norm", type=bool, default=False)

    parser.add_argument("--use_data_augmentation", type=bool, default=False)
    parser.add_argument("--freeze_features", type=bool, default=False)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0)

    parser.add_argument("-r", "--results_file", type=str, default=f"resultados_{datetime.now().strftime('%Y%m%d')}.xlsx")
    args = parser.parse_args()
    main(args.model_name, args.epochs, args.batch_size, 
         args.img_size, 
         args.learning_rate, args.weight_decay,
         args.use_dropout, args.dropout_rate, args.use_batch_norm,
         args.use_data_augmentation, args.freeze_features, args.unfreeze_last_n_layers,         
         args.results_file)



