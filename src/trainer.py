import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import transformer as transformer
from models import transformer_mask
from models import transformer_MoE
from models import sqsformer
from models import conv1d
from models import conv2d
from models import conv3d
from models import SSFTTnet
from models import CASST
from models import SSRN
import utils
from models import transformer_base
from utils import recorder
from evaluation import HSIEvaluation
import itertools
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import device


class SKlearnTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.evalator = HSIEvaluation(param=params)


        self.model = None
        self.real_init()

    def real_init(self):
        pass
        

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY)
        print(self.model, "trian done.") 


    def final_eval(self, testX, testY):
        predictY = self.model.predict(testX)
        temp_res = self.evalator.eval(testY, predictY)
        print(temp_res['oa'], temp_res['aa'], temp_res['kappa'])
        return temp_res

    def test(self, testX):
        return self.model.predict(testX)

            
class SVMTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super(SVMTrainer, self).__init__(params)

    def real_init(self):
        kernel = self.net_params.get('kernel', 'rbf')
        gamma = self.net_params.get('gamma', 'scale')
        c = self.net_params.get('c', 1)
        self.model = svm.SVC(C=c, kernel=kernel, gamma=gamma)

class RandomForestTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n_estimators = self.net_params.get('n_estimators', 200)
        self.model = RandomForestClassifier(n_estimators = n_estimators, max_features="auto", criterion="entropy")

class KNNTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n = self.net_params.get('n', 10)
        self.model = KNeighborsClassifier(n_neighbors=n)

class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device 
        self.evalator = HSIEvaluation(param=params)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 15
        self.unlabel_loader=None
        self.real_init()

    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
       
    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        max_oa = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 60 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                max_oa = max(max_oa, temp_res['oa'])
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                recorder.append_index_value("max_oa", epoch+1, max_oa)
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        y_pred_test, y_test = self.test(test_loader)
        temp_res = self.evalator.eval(y_test, y_pred_test)
        return temp_res

    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            logits = self.get_logits(self.net(inputs))
            if len(logits.shape) == 1:
                continue
            outputs = np.argmax(logits.detach().cpu().numpy(), axis=1)
            
            # ------
            # outputs shape [batch], logits: [batch, class_num]
            # p = torch.softmax(logits, dim=-1)
            # for k, temp_label in enumerate(labels):
            #     if temp_label != outputs[k]:
            #         pre = outputs[k]
            #         real = temp_label.detach().cpu().numpy()
            #         pp = p[k].detach().cpu().numpy()
            #         print("real=%s, %.3f, pred=%s, %.3f" % (real, pp[real], pre, pp[pre]))

            # ------

            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test

class MoETrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super(MoETrainer, self).__init__(params)
        self.task_num = self.net_params.get('task_num', 3)
        self.task_weight = self.net_params.get('task_weight', [1,0,0])
        print("task_weight=", self.task_weight)
        # self.task_label_list = [[1,2,3],[9,10,11]]
        self.task_label_list = self.net_params.get('task_label_list', [[1,2,3], [9,10,11]])
        self.criterion = [nn.CrossEntropyLoss() for i in range(self.task_num)]

    def real_init(self):
        # net
        self.net = transformer_MoE.TransFormerNet(self.params).to(self.device)
        # loss
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        loss = 0
        w = self.task_weight
        i = 0
        for i in range(self.task_num): 
            logit, label = outputs[i], target[i]
            temp_loss = (w[i] * nn.CrossEntropyLoss(ignore_index=-1)(logit, label))
            # print(i, temp_loss)
            loss += temp_loss
        return loss   

    def update_labels(self, labels):
        # labels shape : [batch]
        res = []
        for task_label in self.task_label_list:
            temp_label = labels
            temp_check_label = torch.zeros_like(labels).bool() 
            for i, cur_label in enumerate(task_label):
                temp_label = temp_label.masked_fill(labels==cur_label, i)
                temp_check_label = temp_check_label | (labels==cur_label)
            temp_label = temp_label.masked_fill(temp_check_label==False, -1)     
            res.append(temp_label)
        return res

    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                # print(target.max())
                label_list = [target] + self.update_labels(target)
                data = data.to(self.device) # data: [*], target: [batch]
                labels = [x.to(self.device) for x in label_list]
                outputs = self.net(data) #shape [[batch, class_num0], [batch, class_num1]]

                loss = self.get_loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        y_pred_test, y_test = self.test(test_loader)
        temp_res = self.evalator.eval(y_test, y_pred_test)
        return temp_res

    def get_logits(self, output):
        if type(output) == tuple or type(output) == list:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            logits = self.get_logits(self.net(inputs))
            if len(logits.shape) == 1:
                continue
            outputs = np.argmax(logits.detach().cpu().numpy(), axis=1)
            
            # ------
            # outputs shape [batch], logits: [batch, class_num]
            # p = torch.softmax(logits, dim=-1)
            # for k, temp_label in enumerate(labels):
            #     if temp_label != outputs[k]:
            #         pre = outputs[k]
            #         real = temp_label.detach().cpu().numpy()
            #         pp = p[k].detach().cpu().numpy()
            #         print("real=%s, %.3f, pred=%s, %.3f" % (real, pp[real], pre, pp[pre]))

            # ------

            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test

class SQSformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(SQSformerTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = sqsformer.TransFormerNet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   

class MaskformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(MaskformerTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = transformer_mask.TransFormerNet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            outputs = self.get_logits(self.net(inputs)) # shape = [batch, mask_num, class_num]
            if len(outputs.shape) == 1:
                continue
            
            logits = outputs.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=-1)
            # print(temp_outputs)
            # outputs = []
            # for row in temp_outputs:
                # unique_values, counts = np.unique(row, return_counts=True)
                # most_frequent_number = unique_values[np.argmax(counts)]
                # outputs.append(most_frequent_number)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test


    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   

class TransformerBaseLineTrainer(BaseTrainer):
    def __init__(self, params):
        super(TransformerBaseLineTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = transformer_base.TransFormerNet(self.params).to(self.device)
        # self.net = cross_without_all.HSINet(self.params).to(self.device)
        # self.net = cross_without_center.HSINet(self.params).to(self.device)
        # self.net = cross_without_rotate.HSINet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   




class TransformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(TransformerTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = transformer.TransFormerNet(self.params).to(self.device)
        # self.net = cross_without_all.HSINet(self.params).to(self.device)
        # self.net = cross_without_center.HSINet(self.params).to(self.device)
        # self.net = cross_without_rotate.HSINet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   


class Conv1dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv1d.Conv1d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class Conv2dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv2d.Conv2d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class Conv3dTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = conv3d.Conv3d(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)



class SSFTTTrainer(BaseTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)


    def real_init(self):
        # net
        self.net = SSFTTnet.SSFTTnet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class SSRNTrainer(BaseTrainer):
    def __init__(self, params):
        super(SSRNTrainer, self).__init__(params)

    def get_loss(self, outputs, target):
        logits = outputs
        if len(logits.shape) == 1:
            logits = torch.unsqueeze(logits, 0)
        return self.criterion(logits, target)

    def real_init(self):
        # net
        self.net = SSRN.SSRN(self.params).to(self.device)
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class CASSTTrainer(BaseTrainer):
    def __init__(self, params):
        super(CASSTTrainer, self).__init__(params)

    def get_loss(self, outputs, target):
        logits, A, B = outputs
        return self.criterion(logits, target)

    def real_init(self):
        # net
        self.net = CASST.CASST(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = torch.optim.SGD(self.net.parameters(), 0.005, momentum=0.9, weight_decay=1e-4, nesterov=True)

def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "transformer_base":
        return TransformerBaseLineTrainer(params)
    if trainer_type == "transformer":
        return TransformerTrainer(params)
    if trainer_type == "MoE":
        return MoETrainer(params)
    if trainer_type == "sqsformer":
        return SQSformerTrainer(params)
    if trainer_type == "maskformer":
        return MaskformerTrainer(params)
    if trainer_type == "conv1d":
        return Conv1dTrainer(params)
    if trainer_type == "conv2d":
        return Conv2dTrainer(params)
    if trainer_type == "conv3d":
        return Conv3dTrainer(params)
    if trainer_type == "svm":
        return SVMTrainer(params) 
    if trainer_type == "random_forest":
        return RandomForestTrainer(params)
    if trainer_type == "knn":
        return KNNTrainer(params)
    if trainer_type == "ssftt":
        return SSFTTTrainer(params)
    if trainer_type == "casst":
        return CASSTTrainer(params)
    if trainer_type == "SSRN":
        return SSRNTrainer(params)


    assert Exception("Trainer not implemented!")

