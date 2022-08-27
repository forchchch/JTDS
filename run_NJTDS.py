import torch
from Bird_dataset import Bird_dataset_naive
from torchvision import transforms
from torch.utils.data import DataLoader
from predict_model import Prediction_model
from auxilearn.hypernet import Naive_hyper
from auxilearn.optim import MetaOptimizer
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import os
import random
import numpy as np


########implicit: main_lr:1e-4 aux_lr:1e-3
########one-step: main_lr:0.1  aux_lr:1e-5
parser = argparse.ArgumentParser()
parser.add_argument('--in_dim', type=int, default=512, help='feature dimension')
parser.add_argument('--task_num', type=int, default=1, help='number of tasks')
parser.add_argument('--backbone_pretrain', type=bool, default= True, help='use pretrained backbones')
parser.add_argument('--epochs', type= int , default= 20, help='training epochs')
parser.add_argument('--exp_name', type= str , default= "train_full", help='training epochs')
parser.add_argument('--train_batch_size', type= int , default= 32, help='training batch size')
parser.add_argument('--test_batch_size', type= int , default= 128, help='test batch size')
parser.add_argument('--whether_aux', type=int , default= 0, help='whether use auxiliary tasks')
parser.add_argument('--main_lr', type= float , default= 1e-4, help='main learning rate')
parser.add_argument('--aux_lr', type= float , default= 1e-3, help='aux learning rate')
parser.add_argument('--w_decay', type= float , default= 5e-4, help='weight decay')
parser.add_argument('--auxw_decay', type= float , default= 1e-5, help='aux weight decay')
parser.add_argument('--aux_weight', type= float , default= 1.0, help='aux weight')
parser.add_argument('--aux_batch_size', type= float , default= 50, help='weight decay')
parser.add_argument('--hyperstep', type= int , default= 20, help='step num for aux model')
parser.add_argument('--aux_hidden_dim', nargs='+', type=int, default=[10,10,10,10], help="List of hidden dims for nonlinear")
parser.add_argument('--n_meta_loss_accum', type = int, default = 1, help="accumulated batch number for meta test")
parser.add_argument('--n_meta_train_loss_accum', type = int, default = 1, help="accumulated training batch number for meta test")
parser.add_argument('--corupted', type = int , default= 0, help='noisy setting')
parser.add_argument('--corupted_rate', type = float , default= 0.2, help='noisy setting')
args = parser.parse_args()
print("whether aux:", args.whether_aux)

rand_seed = 7
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
random.seed(rand_seed)
np.random.seed(rand_seed)
#torch.backends.cudnn.deterministic = True

indim = args.in_dim
hyperstep = args.hyperstep
task_num = args.task_num
backbone_pretrain = args.backbone_pretrain
epochs = args.epochs
setting = 'full'
if args.corupted == 1:
    setting = 'noisy'

checkpoint_dir_root = os.path.join('./bird_record',setting,'checkpoint')
log_dir_root = os.path.join('./bird_record',setting,'log')

if not os.path.exists(checkpoint_dir_root):
    os.makedirs(checkpoint_dir_root)
if not os.path.exists(log_dir_root):
    os.makedirs(log_dir_root)

checkpoint_dir = os.path.join(checkpoint_dir_root, args.exp_name)
log_dir = os.path.join(log_dir_root, args.exp_name)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class_num_list = [200]
if task_num > 1:
    for m in range(1,task_num):
        class_num_list.append(2)

logging.basicConfig(level = logging.INFO, filename = os.path.join(log_dir,"log.txt"), filemode = 'w')
logger = logging.getLogger(__name__)
logger.info(args)

def obtain_loss_vector(loss_list):
    return torch.stack( loss_list, dim=1 )

def obtain_loss_matrix(loss_list):
    return torch.stack( loss_list, dim=0 )

def evaluate_model(model,test_loader,device):
    model.eval()
    t_correct = 0
    t_total = 0
    for data in test_loader:
        picture, label, auxlabel= data
        model.to(device)
        picture = picture.to(device)
        label = label.to(device)
        logits,_ = model(picture)
        main_task_logits = logits[0]
        _,predictions = torch.max(main_task_logits,1)
        correct_num = torch.sum(predictions == label) + 0.0
        total_num = len(label)
        t_correct += correct_num
        t_total += total_num
    acc = t_correct/t_total
    model.train()
    return acc

def save_model(model,opt):
    state = {'model':model.state_dict(),
            'opt':opt.state_dict()}
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(checkpoint_dir,'best_predictor.pth'))
    return

def save_bilevel_model(p_model,aux_model,p_opt,aux_opt):
    state = {'p_model':p_model.state_dict(),
            'aux_model':aux_model.state_dict(),
            'p_opt':p_opt.state_dict(),
            'aux_opt':aux_opt.meta_optimizer.state_dict()
            }
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(checkpoint_dir,'best_predictor.pth'))
    return

def save_one_step_model(p_model,aux_model,p_opt,aux_opt):
    state = {'p_model':p_model.state_dict(),
            'aux_model':aux_model.state_dict(),
            'p_opt':p_opt.state_dict(),
            'aux_opt':aux_opt.state_dict()
            }
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(checkpoint_dir,'best_predictor.pth'))
    return
def load_model_test(model,load_dir,test_loader,device):
    info = torch.load(os.path.join(load_dir,'best_predictor.pth'))
    model.load_state_dict(info['p_model'])
    test_acc = evaluate_model(model, test_loader, device)
    return test_acc

def train_meta_model(p_model, aux_model, epochs, train_loader, val_loader, aux_loader, p_opt, aux_opt, criterion,device, task_num, hyperstep):
    p_model = p_model.to(device)
    aux_model = aux_model.to(device)
    p_model.train()
    aux_model.train()
    counter = 0
    best_acc = 0.0
    for epoch in range(epochs):
        #running_loss = 0.0
        ite = 0
        for data in train_loader:
            picture, label, auxlabel,  data_id = data
            data_id = data_id.to(device)
            picture = picture.to(device)
            label = label.to(device)
            for m in range(1,task_num):
                auxlabel[m-1] = auxlabel[m-1].to(device)
            #auxlabel = auxlabel.to(device)
            logits,_ = p_model(picture)
            p_opt.zero_grad()
            main_task_logits = logits[0]
            loss_list = []
            main_loss = criterion(main_task_logits,label)
            loss_list.append(main_loss)
            for m in range(1,task_num):
                loss_list.append(args.aux_weight*criterion(logits[m],auxlabel[m-1]))
            loss_vector = obtain_loss_vector(loss_list)
            total_loss = aux_model(loss_vector, data_id)
            total_loss.backward()
            p_opt.step()

            running_loss = total_loss.item()
            target_loss = main_loss.mean().item()
            _,predictions = torch.max(main_task_logits,1)
            correct_num = torch.sum(predictions == label) + 0.0
            total_num = len(label)

            #exit(0)
            if (counter+1) % hyperstep == 0:
                #print("start here",counter)
                meta_val_loss = 0.0
                for n_val_step, aux_data in enumerate(aux_loader):
                    if n_val_step < args.n_meta_loss_accum:
                        aux_picture, aux_label, aux_auxlabel,_ = aux_data
                        aux_picture = aux_picture.to(device)
                        aux_label = aux_label.to(device)
                        for m in range(1,task_num):
                            aux_auxlabel[m-1] = aux_auxlabel[m-1].to(device)

                        #auxlabel = auxlabel.to(device)
                        aux_logits,_ = p_model(aux_picture)
                        main_task_logits = aux_logits[0]
                        main_loss = criterion(main_task_logits,aux_label).mean()
                        for m in range(1,task_num):
                            main_loss += 0.0*criterion(aux_logits[m],aux_auxlabel[m-1]).mean()
                        meta_val_loss += main_loss 
                    else:
                        break

                # inner_loop_end_train_loss, e.g. dL_train/dw
                total_meta_train_loss = 0.
                for n_train_step, data in enumerate(train_loader):
                    if n_train_step < args.n_meta_loss_accum:
                        picture, label, auxlabel,  data_id = data
                        picture = picture.to(device)
                        data_id = data_id.to(device)
                        label = label.to(device)
                        for m in range(1,task_num):
                            auxlabel[m-1] = auxlabel[m-1].to(device)
                        #auxlabel = auxlabel.to(device)
                        logits,_ = p_model(picture)
                        main_task_logits = logits[0]
                        loss_list = []
                        main_loss = criterion(main_task_logits,label)
                        loss_list.append(main_loss)
                        for m in range(1,task_num):
                            loss_list.append(args.aux_weight*criterion(logits[m],auxlabel[m-1]))
                        loss_vector = obtain_loss_vector(loss_list)
                        total_loss = aux_model(loss_vector, data_id)
                        total_meta_train_loss += total_loss
                    else:
                        break

                print( "epoch:", epoch, "ite:", ite, "meta_loss:", meta_val_loss.item(), "combined_loss:", total_meta_train_loss.item() )
                logger.info( "epoch: %d, ite: %d, meta_loss: %.6f, combined_loss:%.6f "%(epoch, ite, meta_val_loss.item(), total_meta_train_loss.item()) )
                aux_opt.step(
                    val_loss=meta_val_loss,
                    train_loss=total_meta_train_loss,
                    aux_params = list(aux_model.parameters()),
                    parameters = list(p_model.parameters())
                )
            if (ite+1) % 20 == 0:
                print("epoch",epoch,"ite:",ite,"training loss:",running_loss,"main task loss:",target_loss, "training acc:",correct_num/total_num)
                logger.info("epoch:%d ,iteration:%d, training loss:%.6f, main task loss:%.6f, training acc:%.6f" %(epoch,ite,running_loss,target_loss,correct_num/total_num) )
            ite += 1
            counter += 1
        acc = evaluate_model(p_model,val_loader,device)
        if acc>best_acc:
            best_acc = acc
            save_bilevel_model(p_model,aux_model,p_opt,aux_opt)
        print("current accuacy:",acc,"best accuracy:",best_acc)
        logger.info("current accuacy: %.6f, best accuracy: %.6f "%(acc,best_acc) )
    return

train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

train_image_file = './preprocess_data/rest_train_set.json'
valid_image_file = './preprocess_data/valid_set.json'
auxilary_image_file = './preprocess_data/aux_set.json'
test_image_file = './preprocess_data/test_set.json'
label_file = './preprocess_data/image_dictionary.json'
image_root = '/DATA/DATANAS1/chenhong/select_task_dataset/CUB_200_2011/images'
#image_root = './CUB_200_2011/images'

train_dataset = Bird_dataset_naive(train_image_file,label_file,image_root,transform=train_transform,finegrain=args.whether_aux,corrupted=args.corupted, cor_rate= args.corupted_rate)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,drop_last=False)

valid_dataset = Bird_dataset_naive(valid_image_file,label_file,image_root,transform=test_transform,finegrain=False)
valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False,drop_last=False)

test_dataset = Bird_dataset_naive(test_image_file,label_file,image_root,transform=test_transform,finegrain=False)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,drop_last=False)

aux_dataset = Bird_dataset_naive(auxilary_image_file,label_file,image_root,transform=test_transform,finegrain=True)
aux_loader = DataLoader(aux_dataset, batch_size=args.aux_batch_size, shuffle=False,drop_last=False)

#mymodel = Base_models.resnet18(pretrained = True)
data_num = len(train_dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
p_model = Prediction_model(task_num, indim, class_num_list, backbone_pretrain)
aux_model = Naive_hyper(data_num=data_num, task_num = task_num)
crit = nn.CrossEntropyLoss(reduction='none')

optimizer_main = optim.Adam(p_model.parameters(), lr=args.main_lr, weight_decay=args.w_decay)
optimizer_aux = optim.SGD( aux_model.parameters(), lr=args.aux_lr, momentum=0.9, weight_decay=args.auxw_decay )
#optimizer_aux = optim.Adam( aux_model.parameters(), lr=args.aux_lr, weight_decay=args.auxw_decay )
meta_optimizer = MetaOptimizer(
    meta_optimizer= optimizer_aux,
    hpo_lr = args.main_lr,
    truncate_iter = 3,
    max_grad_norm = 10 
)
train_meta_model(p_model, aux_model, epochs, train_loader, valid_loader, aux_loader, optimizer_main, meta_optimizer, crit, device, task_num, hyperstep)
test_acc = load_model_test(p_model,checkpoint_dir,test_loader, device)
print("test accuracy:", test_acc)
logger.info("best test accuacy: %.6f"%(test_acc) )



