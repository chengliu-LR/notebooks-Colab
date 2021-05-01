import torch
import torch.nn as nn
from collections import OrderedDict

seq_model = nn.Sequential(
            nn.Linear(1, 11),
            nn.Tanh(),
            nn.Linear(11, 1)
            )

namedseq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 13)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(13, 1))
    ]))

class SelfSubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 8)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(8, 1)
        #assigning an instance of nn.Module to an attribute in a nn.Module, as you did in the constructor here, automatically registers the module as a submodule, which gives modules(SelfSubModule) access to the parameters of its submodules(hidden_linear, hidden_activation, output_linear) without further action by the user.
    def forward(self, input):   #input is the input data
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t

subclass_model = SelfSubModule()
#what happens below is that the named_parameters() call delves into all submodules assigned as attributes in the constructor and recursively calls named_parameters() on each one of them.
for type_str, model in [('seq', seq_model), ('namedseq', namedseq_model), ('subclass', subclass_model)]:
    print(type_str)
    for name_str, param in model.named_parameters():
        print("{:21} {:19}".format(name_str, str(param.shape)))
    print()

print('hidden_activation:', subclass_model.hidden_activation) ##The Tanh() module can be accessed as an attribute using the given name:"hidden_activation".
