import numpy as np
import torch
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

class BaseDataset:
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, x, y) -> None:
        self.X = x
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple:
        return self.X[idx], self.y[idx]
        

class BaseDatastreamer:
    """This datastreamer wil never stop
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(
        self,
        dataset,
        batchsize,
        preprocessor = None,
        random = True,
    ) -> None:
        self.dataset = dataset
        self.batchsize = batchsize
        self.preprocessor = preprocessor
        self.random = random
        self.size = len(self.dataset)
        self.reset_index()

    def __len__(self):
        return int(len(self.dataset) / self.batchsize)

    def reset_index(self) -> None:
        if self.random:
            self.index_list = np.random.permutation(self.size)
        else:
            self.index_list =np.arange(self.size)
        self.index = 0

    def batchloop(self):
        batch = []
        for _ in range(self.batchsize):
            x = self.dataset[int(self.index_list[self.index])]
            batch.append((x))
            self.index += 1
        return batch

    def stream(self):
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            if self.preprocessor is not None:
                X, Y = self.preprocessor(batch)  # noqa N806
            else:
                X, Y = zip(*batch)  # noqa N806
                X, Y = torch.stack(X), torch.stack(Y)
                Y = torch.unsqueeze(Y,1)
            yield X, Y



class VAEstreamer(BaseDatastreamer):

    def __init__(
            self,
            dataset,
            batchsize,
            # one_hot_enc=None,
            # cols_to_encode=None,
            loss_func_name,
            preprocessor = None,
            random = True,
    ):
        super().__init__(dataset,batchsize,preprocessor,random)
        # self.cols_to_encode = cols_to_encode
        self.loss_func_name = loss_func_name
        # self.one_hot_enc = one_hot_enc
    

    # def one_hot_encode(self,tensor,cols_to_encode):
    #     if type(cols_to_encode) != type([]):
    #         print("no_list")
    #         cols_to_encode = [cols_to_encode]

    #     one_hot_tensor = torch.zeros(tensor[:,:,:0].shape)
    #     for pos in sorted(cols_to_encode, reverse=True):
            
    #         num_classes = tensor[:,:,pos].long().max().item() + 1 #heeft one_hot_encoder nodig, eiglijk zijn het num_calses-1
    #         one_hot_column = torch.nn.functional.one_hot(tensor[:,:,pos].long(), num_classes=num_classes)[:,:,1:] # remove the extra 0 column
            
    #         if self.one_hot_enc == "only_one_hot":
    #             # get only one hot encoding 
    #             # for pure casification
    #             one_hot_tensor = torch.cat((one_hot_column,one_hot_tensor), dim=2)

    #         elif self.one_hot_enc == "combined":
    #             # get one hot enciding concat to original
    #             # for calsification + regression
    #              tensor = torch.cat((tensor[:,:,:pos], one_hot_column, tensor[:,:,pos+1:]), dim=2)
            
           
    #     if self.one_hot_enc == "only_one_hot":
    #         return one_hot_tensor
    #     else:
    #         return tensor
        


    def stream(self):
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            # we throw away the Y
            X_ = batch
            
            X = torch.stack(X_)
            
            if self.loss_func_name == "RMSE":
                Y = X[:,:,:-1].float()

            elif self.loss_func_name == "CEL":
                Y = X[:,:,-1:].squeeze(-1).long()
            
            elif self.loss_func_name == "KLD":
                
                Y = X[:,:,-1:].squeeze(-1).long()
                # X = X[:,:,-1].unsqueeze(-1).long() # ONLY categories

            else:
                Y = X

            # # cols_to_encode = [6,7]
            # if self.one_hot_enc is "original_data": # self.cols_to_encode is None:
            #     # get only orginal data 
            #     # for regression
            #     Y = X
            # elif self.one_hot_enc in ["only_one_hot","combined"]:
            #     # get only one hot encoding or and concat to original
            #     # for pure casification or calsification + regression
            #     Y = self.one_hot_encode(X, self.cols_to_encode)
            
         
            yield X, Y