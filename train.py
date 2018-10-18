import copy
import time

import torch
import torch.nn as nn   
import torch.optim as optim
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt

import rdataset
import unet


class trainResults():
    '''
    Guarda resultados de treinamento gerados pela função train_model
    '''
    def __init__(self, best_model, vlh, vah, tlh, tah, nepochs):
        self.best_model = best_model
        self.val_loss_hist = vlh
        self.val_acc_hist = vah
        self.train_loss_hist = tlh
        self.train_acc_hist = tah
        self.nepochs = nepochs

    def plot(self):
        epoch_range = range(1, self.nepochs + 1)
        plt.subplot(1,2,1)
        plt.ylabel("DICE")
        plt.xlabel("Epoch")
        plt.plot(epoch_range, self.val_acc_hist, 'r*-', label='Val DICE')
        plt.legend()
        plt.plot(epoch_range, self.train_acc_hist, 'b*-', label='Train DICE')
        plt.legend()
        plt.subplot(1,2,2)
        plt.ylabel("BCELoss")
        plt.xlabel("Epoch")
        plt.plot(epoch_range, self.val_loss_hist, 'r*-', label='Validation')
        plt.legend()
        plt.plot(epoch_range, self.train_loss_hist, 'b*-', label='Train')
        plt.legend()
        plt.show()
        
        
def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=10):
    '''
    Treina um modelo utilizando loss e otimizador passados
    '''
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    val_loss_hist = []
    val_acc_hist = []
    train_loss_hist = []
    train_acc_hist = []
    sigmoid = nn.Sigmoid()
    
    # Para cada epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        loss_por_batch = []
        # Cada epoch realiza treino e validação
        for phase in ['train', 'validation']: # validação desabilitada
            if phase == 'train':
                scheduler.step() # possivelmente varia learning rate, dependendo de configuração externa
                model.train()  # Modo de treinamento
            else:
                model.eval()   # Modo de validação

            running_loss = 0.0
            acc = 0.0

            # Dentro de um epoch, iterar sobre os batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zera gradientes
                optimizer.zero_grad()

                # Passa o batch pelo modelo
                # Calcula a perda
                # Só liga a computação do grafo computacional quando na fase de treinamento
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backpropagation e otimiza quando em treinamento
                    if phase == 'train':
                        loss_por_batch.append(loss)
                        loss.backward()
                        optimizer.step()
                        
                        
                # Computa estatisticas do batch
                running_loss += loss.item() * inputs.size(0)
                acc += torch.abs(outputs - labels)
                
            # Depois que passa por todos os dados, statisticas do epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = acc / dataset_sizes[phase]
            
            # Salvando estatisticas
            if phase == 'train':
                train_loss_hist.append(epoch_loss)
                train_acc_hist.append(epoch_acc)
                plt.title("Training Loss por batch, EPOCH: " + str(epoch + 1)) # inicialização de gráfico de loss
                plt.xlabel('Batch')
                plt.ylabel('MSE Loss')
                plt.plot(range(len(loss_por_batch)), loss_por_batch, 'b*-', label='Train')
                plt.show()
            elif phase == 'validation':
                val_loss_hist.append(epoch_loss)
                val_acc_hist.append(epoch_acc) 
            else:
                raise ValueError("WRONG PHASE NAME, check folder names?")
            
            print('{} BCELoss: {:.4f} DICE: {:.4f} '.format(
                phase, epoch_loss, epoch_acc))

            # Mantem copia do melhor modelo (de treino por enqnto)
            if phase == 'validation' and epoch_acc > best_acc:
                print("Best model so far, checkpoint...")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:4f}'.format(best_acc))

    # Retorna melhor modelo
    model.load_state_dict(best_model_wts)
    results = trainResults(model, val_loss_hist, val_acc_hist,
                           train_loss_hist, train_acc_hist, num_epochs)
    return results


def test_model(model, criterion, dataloaders, phase="test"):
    '''
    Treina um modelo utilizando loss e otimizador passados
    '''
    
    since = time.time()
    
    sigmoid = nn.Sigmoid()
    
    print("Running test in " + phase + " dataloader.")
    model.eval()   # Modo de validação

    running_loss = 0.0
    acc = 0.0

    # Dentro de um epoch, iterar sobre os batches
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            soutput = sigmoid(outputs)
            mask_output = ((soutput - soutput.min())/soutput.max() > 0.5).float()
            dice = batch_dice(mask_output, labels)

        # Computa estatisticas do batch
        running_loss += loss.item() * inputs.size(0)
        acc += dice * inputs.size(0) 

    # Depois que passa por todos os dados, statisticas do epoch
    test_loss = running_loss / dataset_sizes[phase]
    test_acc = acc / dataset_sizes[phase]
    

    print('{} BCELoss: {:.4f} DICE: {:.4f} '.format(
        phase, test_loss, test_acc))

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


rec_dataloaders, device, dataset_sizes = rdataset.prepare_environment()
unet = UNet(1, 1)  # 1 channel (grayscale), 1 class (hippocampus)
torch.cuda.empty_cache()
unet = unet.to(device)

criterion = nn.BCEWithLogitsLoss()
opt = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9) # tentando sgd
scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.5) # half learning rate every 10 epochs
model_name = "unet-test-axial"
