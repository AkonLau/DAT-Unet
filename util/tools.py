import random
import numpy as np
import torch
import copy

class SeedContextManager:
    def __init__(self, seed=None):
        self.seed = seed + 40 if seed is not None else None
        self.random_state = None
        self.np_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        if self.seed is not None:
            self.random_state = random.getstate()
            self.np_random_state = np.random.get_state()
            self.torch_random_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                self.torch_cuda_random_state = torch.cuda.random.get_rng_state_all()

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is not None:
            random.setstate(self.random_state)
            np.random.set_state(self.np_random_state)
            torch.random.set_rng_state(self.torch_random_state)
            if torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(self.torch_cuda_random_state)


def load_from(model, pretrained_path):
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model" not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            model.load_state_dict(pretrained_dict, strict=False)
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained model of swin encoder---")

        model_dict = model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3 - int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k: v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                # print(k)
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                    del full_dict[k]

        model.load_state_dict(full_dict, strict=False)
    else:
        print("none pretrain")

