# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler


def save_checkpoint(state, best_accuracy, save_dir):
    print("Saving a new best model")
    checkpoint_path = "checkpoint_acc_{:.2f}.pth".format(best_accuracy*100)
    checkpoint_path = save_dir + checkpoint_path
    torch.save(state, checkpoint_path)


def save_pred_checkpoint(state, best_accuracy):
    print("Saving a new best model")
    checkpoint_path = "checkpoint_pred_acc_{:.2f}.pth".format(best_accuracy*100)
    torch.save(state, checkpoint_path)

# train / eval

def fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    best_accuracy = 0
    best_loss = 5
    num = 0
    for epoch in range(num_epochs):
        model.train()

        for i, data in enumerate(train_loader):
            audio = data['mel']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            if args.gpu_use == 1:
                audio = Variable(audio).type(torch.FloatTensor).cuda(args.which_gpu)
                label = Variable(label).type(torch.LongTensor).cuda(args.which_gpu)
            elif args.gpu_use == 0:
                audio = Variable(audio).type(torch.FloatTensor)
                label = Variable(label).type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(audio)

            #print(outputs,label)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))
        """"
        eval_loss, output_all, label_all = eval(model, valid_loader, criterion, args)
        prediction = np.concatenate(output_all)
        prediction = prediction.reshape([len(prediction)/9, 9, 10])
        prediction = np.mean(prediction, axis=1)
        prediction = prediction.argmax(axis=1)
        y_label = np.concatenate(label_all)
        y_label = y_label[::9]

        comparison = prediction - y_label
        acc = float(len(comparison) - np.count_nonzero(comparison)) / len(comparison)
        print(acc)
        """
        #if acc >= 0.60 :
            #num = 0
            #best_accuracy = acc
            #best_loss = eval_loss
        #    break
        #else:
        #    num += 1
        #np.maximum(acc, best_accuracy.numpy())
        #best_accuracy = torch.from_numpy(best_accuracy)
        #print(num)
        eval_loss, _, _ = eval(model, valid_loader, criterion, args)
        scheduler.step(eval_loss) # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-6:
            print ("Early stopping\n\n")
            break
        #if num == 20 :
        #    print ("Early stopping\n\n")
        #    break


def fit_predict(model, train_loader, valid_loader, criterion, learning_rate, num_epochs, args):
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, verbose=True)
    best_loss = 5.0
    num = 0
    for epoch in range(num_epochs):
        model.train()

        for i, data in enumerate(train_loader):
            audio = data['mel']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            if args.gpu_use == 1:
                audio = Variable(audio).type(torch.FloatTensor).cuda(args.which_gpu)
                label = Variable(label).type(torch.LongTensor).cuda(args.which_gpu)
            elif args.gpu_use == 0:
                audio = Variable(audio).type(torch.FloatTensor)
                label = Variable(label).type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(audio)

            # print(outputs,label)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (
                    epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0]))

        eval_loss, output_all, label_all = eval(model, valid_loader, criterion, args)

        if best_loss >= eval_loss :
            num = 0
            #best_accuracy = acc
            best_loss = eval_loss

        else:
            num += 1

        scheduler.step(eval_loss) # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        #if curr_lr < 1e-8:
        #    print ("Early stopping\n\n")
        #    break
        if num == 20 :
            print ("Early stopping\n\n")
            break

def eval(model,valid_loader,criterion, args):

    eval_loss = 0.0
    output_all = []
    label_all = []

    model.eval()
    for i, data in enumerate(valid_loader):
        audio = data['mel']
        label = data['label']
        # have to convert to an autograd.Variable type in order to keep track of the gradient...
        if args.gpu_use == 1:
            audio = Variable(audio).type(torch.FloatTensor).cuda(args.which_gpu)
            label = Variable(label).type(torch.LongTensor).cuda(args.which_gpu)
        elif args.gpu_use == 0:
            audio = Variable(audio).type(torch.FloatTensor)
            label = Variable(label).type(torch.LongTensor)
	
        outputs = model(audio)
        loss = criterion(outputs, label)

        eval_loss += loss.data[0]

        output_all.append(outputs.data.cpu().numpy())
        label_all.append(label.data.cpu().numpy())

    avg_loss = eval_loss/len(valid_loader)
    print ('Average loss: {:.4f} \n'. format(avg_loss))



    return avg_loss, output_all, label_all

def get_prediction(data, model, criterion, args):
    from torch.utils.data import DataLoader
    loader_prediction = DataLoader(data, batch_size=9, shuffle=False, drop_last=False)
    _, prediction, label = eval(model,loader_prediction,criterion, args)
    prediction = np.asarray(prediction)
    #prediction = np.concatenate(prediction)
    #prediction = prediction.reshape(-1, 10)
    #label = np.concatenate(label)
    label = np.asarray(label)
    prediction = np.transpose(prediction, [0, 2, 1])
    label = label[:,0]

    return prediction, label




