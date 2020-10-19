import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
import numpy as np
from Net import SeG
from sklearn import metrics
from data_loader import data_loader
from config import config
from utils import AverageMeter

def train(train_loader, test_loader, opt):
    model = SeG(train_loader.dataset.vec_save_dir, train_loader.dataset.rel_num(),
                lambda_pcnn=opt['lambda_pcnn'], lambda_san=opt['lambda_san'])
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight())
    optimizer = optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=1e-5)
    not_best_count = 0
    best_auc = 0
    best_model = None
    for epoch in range(opt['epoch']):
        model.train()
        print("=== Epoch %d train ===" % epoch)
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()
        for i, data in enumerate(train_loader):
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda()
            word, pos1, pos2, ent1, ent2, mask, scope, rel = data
            output = model(word, pos1, pos2, ent1, ent2, mask, scope)
            loss = criterion(output, rel)
            _, pred = torch.max(output, -1)
            acc = (pred == rel).sum().item() / rel.shape[0]
            pos_total = (rel != 0).sum().item()
            pos_correct = ((pred == rel) & (rel != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | loss: %f, acc: %f, pos_acc: %f'%(i+1, avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            # Optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % opt['val_iter'] == 0:
            print("\n=== Epoch %d val ===" % epoch)
            y_true, y_pred = valid(test_loader, model)
            auc = metrics.average_precision_score(y_true, y_pred)
            print("\n[TEST] auc: {}".format(auc))
            if auc > best_auc:
                print("Best result!")
                best_auc = auc
                best_model = model
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= opt['early_stop']:
                break
    return best_model


def valid(test_loader, model):
    model.eval()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            word, pos1, pos2, ent1, ent2, mask, scope, rel = data
            output = torch.softmax(model(word, pos1, pos2, ent1, ent2, mask, scope), -1)
            label = rel.argmax(-1)
            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f'%(i+1, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])
    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    return y_true, y_pred

def test(test_loader, model):
    y_true, y_pred = valid(test_loader, model)
    auc = metrics.average_precision_score(y_true, y_pred)
    print("\n[TEST] auc: {}".format(auc))
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean()
    p200 = (y_true[order[:200]]).mean()
    p300 = (y_true[order[:300]]).mean()
    print("P@100: {0:.1f}".format(p100*100))
    print("P@200: {0:.1f}".format(p200*100))
    print("P@300: {0:.1f}".format(p300*100))
    print("mean: {0:.1f}".format((p300+p200+p300)/0.03))

if __name__ == '__main__':
    opt = vars(config())
    train_loader = data_loader(opt['train'], opt, shuffle=True, training=True)
    test_loader = data_loader(opt['test'], opt, shuffle=False, training=False)
    best_model = train(train_loader, test_loader, opt)
    test(test_loader, best_model)

