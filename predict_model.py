import Base_models
import torch.nn as nn

class Prediction_model(nn.Module):
    def __init__(self, task_num, indim, class_num_list, backbone_pretrain,data = 'CUB'):
        super(Prediction_model, self).__init__()
        self.task_num = task_num
        if data == "cifar":
            #self.base_model = Base_model2.resnet18(pretrained = backbone_pretrain)
            #self.base_model = Base_models.resnet18(pretrained = True)
            self.base_model = Base_models.Convnet()
        else:
            self.base_model = Base_models.resnet18(pretrained = backbone_pretrain)
        self.head_list = nn.ModuleList()
        for m in range(self.task_num):
            # head = nn.Sequential(
            #     nn.Linear(indim, indim//2),
            #     nn.ReLU(),
            #     nn.Linear(indim//2, indim//2),
            #     nn.ReLU(),
            #     nn.Linear(indim//2 , class_num_list[m])
            # )
            head = nn.Sequential(
                nn.Linear(indim , class_num_list[m])
            )
            self.head_list.append(head)
    
    def forward(self,x):
        feature = self.base_model(x)
        predicted_logits = []
        for m in range(self.task_num):
            logits = self.head_list[m](feature)
            predicted_logits.append(logits)
        return predicted_logits,feature