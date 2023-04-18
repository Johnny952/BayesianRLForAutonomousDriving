import copy
import warnings
import torch

import operator

ops = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : operator.truediv,
    '%' : operator.mod,
    '^' : operator.xor,
}

##### torchhk
def transform_layer(input, from_inst, to_inst, args={}, attrs={}):
    if isinstance(input, from_inst) :
        for key in args.keys() :
            arg = args[key]
            if isinstance(arg, str) :
                if arg.startswith(".") :
                    args[key] = getattr(input, arg[1:])
                    
        output = to_inst(**args)
        
        for key in attrs.keys() :
            attr = attrs[key]
            if isinstance(attr, str):
                if attr.startswith("."):
                    attrs[key] = getattr(input, attr[1:])
            if isinstance(attr, list) and len(attr) == 2:
                if attr[0].startswith(".") and attr[1].startswith("."):
                    attr1 = getattr(input, attr[0][1:])
                    attr2 = getattr(input, attr[1][1:])
                    attrs[key] = torch.nn.Parameter(attr1 + torch.exp(attr2))

            setattr(output, key, attrs[key])
    else :
        output = input        
    return output


def transform_model(input, from_inst, to_inst, args={}, attrs={}, inplace=True, _warn=True):
    if inplace :
        output = input
        if _warn :
            warnings.warn("\n * Caution : The Input Model is CHANGED because inplace=True.", Warning)
    else :
        output = copy.deepcopy(input)
    
    if isinstance(output, from_inst) :
        output = transform_layer(output, from_inst, to_inst, copy.deepcopy(args), copy.deepcopy(attrs))
    else :
        for name, module in output.named_children() :
            setattr(output, name, transform_model(module, from_inst, to_inst, copy.deepcopy(args), copy.deepcopy(attrs), _warn=False))
            
    return output