import time
import torch
import copy

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    now = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        # TODO: change to cli progress bar
        print(f'Epochs {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has training and validating phase
        for phase in ['train', 'val']:

            print(f'Starting {phase} phase')
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate data
            for sample in dataloaders[phase]:
                # computation to GPU
                image = sample["image"]
                label = sample["label"]

                data = image.to(device=device, dtype=torch.float)
                targets = label.to(device)

                # zero the parameter grads
                optimizer.zero_grad()

                # forward
                # track history for train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, targets)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == targets.data) # debug target.data

            if phase == 'train':
                scheduler.step() # check LR scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - now
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))

    # load best model weights
    model.state_dict(best_model)
    return model

