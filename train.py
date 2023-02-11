from pathlib import Path
from collections import defaultdict
from tkinter.messagebox import NO
import numpy as np
import os
import torch
from torch import nn

import gc
import argparse
from torch.utils.data import DataLoader

import model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import roc_curve
import config
import data_utils

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# In[19]:
verbos = 0
checkpoint = 'checkpoint'
model_name = 'best_emodb.pt'
def train(train_loader, valid_loader, valid_count, load, freeze, max_iters):
    y_pred_valid = np.empty((4 * valid_count),dtype=np.float32)
    y_targ_valid = np.empty((4 * valid_count),dtype=np.float32)
    best_valid_uw = 0
    best_valid_ac = 0
    best_eer = 0
    best_eer_thresh = 0
    ##########tarin model###########
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            m.weight.data.normal_(0.0, 0.1)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == torch.nn.Conv2d:
            m.weight.data.normal_(0.0, 0.1)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    model_ = model.model_emo_mh(num_classes=4, Di1=config.timesteps, Di2=512, Drc=128, Fc1=128, Fc2=64)
    model_.apply(init_weights)
    model_ = model_.to(device)

    criterion = nn.BCELoss()
    criterion = criterion.cuda()

    if load:
        model_dict = model_.state_dict()

        state_dict = {}
        criteriondict = {}


        saved_model = torch.load(os.path.join("checkpoint_best_new", "best_model204.pth")) #204
        state_dict = saved_model['state_dict']
        criteriondict = saved_model['classifier']
        criterion.load_state_dict(criteriondict)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}#k not in ['W']}
        model_dict.update(pretrained_dict)
        model_.load_state_dict(model_dict)

    if freeze:
        for param in model_.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(criterion.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = None
    else:
        optimizer = optim.Adam([
            {'params': list(model_.parameters())},
            {'params': list(criterion.parameters()), 'lr': config.learning_rate}],
            lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr = 1e-2, total_steps = max_iters, pct_start = 0.2)

    for epoch in range(config.max_epochs):
        model_.train()
        for train_data_1, train_data_2, train_label in train_loader:
            input1 = train_data_1.to(device)
            input2 = train_data_2.to(device)
            targets = train_label.to(device,torch.float32)
            optimizer.zero_grad()
            outputs = model_(input1, input2)
            loss = criterion(outputs.squeeze(1), targets)
            loss.backward()
            if config.clip:
                torch.nn.utils.clip_grad_norm_(model_.parameters(), config.clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        index = 0
        cost_valid = 0
        model_.eval()
        # valid_data =
        with torch.no_grad():
            for valid_data_1, valid_data_2, Valid_label_ in valid_loader:
                input1 = valid_data_1.to(device)
                input2 = valid_data_2.to(device)
                targets = Valid_label_.to(device, torch.float32)
                outputs = model_(input2, input2).squeeze(1)
                loss = criterion(outputs, targets)
                y_pred_valid[index:index+targets.shape[0]] = outputs.cpu().detach().numpy()
                y_targ_valid[index:index+targets.shape[0]] = Valid_label_
                index += targets.shape[0]
            cost_valid = cost_valid + np.sum(loss.cpu().detach().numpy())
        model_.train()
        index = 0
        cost_valid = cost_valid/len(y_pred_valid)
        fpr, tpr, threshold = roc_curve(y_targ_valid, y_pred_valid, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        y_pred_valid[y_pred_valid < eer_threshold] = 0
        y_pred_valid[y_pred_valid >= eer_threshold] = 1
          # compute evaluated results
        valid_rec_uw = recall(y_pred_valid, y_targ_valid, average='macro')
        valid_conf = confusion(y_pred_valid, y_targ_valid)
        valid_acc_uw = accuracy(y_pred_valid, y_targ_valid)




        if valid_acc_uw > best_valid_ac:
            best_valid_ac = valid_acc_uw
            best_valid_uw = valid_rec_uw
            best_valid_conf = valid_conf
            best_eer_thresh = eer_threshold
            best_eer = EER
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save({"state_dict": model_.state_dict(), "classifier": criterion.state_dict()}, os.path.join(checkpoint, model_name))

        if verbos:
            # print results
            print ("*****************************************************************")
            print ("Epoch: %05d" %(epoch+1))
            # print ("Training cost: %2.3g" %tcost)
            # print ("Training accuracy: %3.4g" %tracc)
            print ("Valid cost: %2.3g" %cost_valid)
            print ("Valid_Recall: %3.4g" %valid_rec_uw)
            print ("Best valid_RE: %3.4g" %best_valid_uw)
            print ("Valid_Accuracy: %3.4g" %valid_acc_uw)
            print ("Best valid_Acc: %3.4g" %best_valid_ac)
            print ('Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print (valid_conf)
            print ('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print (best_valid_conf)
            print ("*****************************************************************" )

    return best_valid_ac, best_eer, best_eer_thresh




#seed = 22
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
def cross_fold(args):
    total_acval = []
    total_erval = []
    total_erthr = []
    for fn, fold in enumerate(range(0,10,2)):
        train_idx = [str(fi_) for fo in [ses[fold], ses[(fold+1)]] for fi_ in fo]
        eval_idx = [str(fi_) for fo in range(10) if fo not in [fold, (fold+1)] for fi_ in ses[fo]]
        train_data, train_label, train_emt = data_utils.read_EMODB(audio_indexes = train_idx, is_training = True, filter_num = config.feat_dim, timesteps = config.timesteps)
        eval_data, eval_label, eval_emt = data_utils.read_EMODB(audio_indexes = eval_idx, is_training = False, filter_num = config.feat_dim, timesteps = config.timesteps)
        train_label = train_label.squeeze(1)
        eval_label = eval_label.squeeze(1)

        batch_size = args.batch_size
        train_dataset = data_utils.data_emo(train_data, train_label.to(torch.long))
        test_dataset = data_utils.data_emo(eval_data, eval_label.to(torch.long), train_data, train_label.to(torch.long))

        emob_sampler = data_utils.balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
            sampler=emob_sampler, num_workers=0, pin_memory=False, drop_last=False)

        emoe_sampler = data_utils.eval_sampler(test_dataset, train_dataset)
        eval_loader = DataLoader(test_dataset, batch_size=batch_size,
            shuffle=False, sampler=emoe_sampler, num_workers=0, pin_memory=False, drop_last=False)
        acc_val, eer_val, eerthresh_val = train(train_loader, eval_loader, valid_count = eval_label.shape[0], load = args.load_model, freeze = args.freeze, max_iters = config.max_epochs * (int(len(train_idx) / batch_size)+1))
        print("Valid_Acc: %3.4g EER: %3.2g in fold %d" %(acc_val, eerthresh_val, fn))
        del train_data, train_label, train_emt
        del eval_data, eval_label, eval_emt
        del train_loader, eval_loader
        total_acval.append(acc_val)
        total_erval.append(eer_val)
        total_erthr.append(eerthresh_val)

        gc.collect()

    print(sum(total_acval)/len(total_acval))
    print(sum(total_erthr)/len(total_erthr))
    print(sum(total_erval)/len(total_erval))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str,
                        required=True, help='path of the data directory where features are extracted into')
    parser.add_argument('-l', '--load_model', type=str,
                        required=False, help='path to load the saved checkpoint')
    parser.add_argument('--freeze', action='store_true',
                        required=False, help='pass it if you want to freeze a part of the model')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        required=False, help="pass batch size for trainin")

    args = parser.parse_args()

    # In[7]:
    data_dir = Path(args.data_dir)
    ses = defaultdict(list)

    for f, g in enumerate(config.speakers):
        ses[f] = list(data_dir.joinpath(g).glob('*.pt'))

    cross_fold(args)