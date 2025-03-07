import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import os


from models.quan_yolo_data_trans import *

flops = {}
params = {}
layer_info = {}

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# quantize_k function in DoReFa-Net (Eq.(5))
class quantize_k(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_i: torch.Tensor, k_bit: int):
        r_o = torch.floor((2**k_bit - 1) * r_i) / (2 **k_bit - 1)
        return r_o
    
    @staticmethod
    def backward(ctx, grad_outputs):
       return grad_outputs.clone(), None
   
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coe: int):
        r_o = torch.floor(coe * inputs) / coe
        return r_o
    
    @staticmethod
    def backward(ctx, grad_outputs):
       return grad_outputs.clone(), None
   
# binary quantization in DoReFa-Net (Eq.(7))
class binary_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r_i: torch.Tensor, e_bit: int):
        r_o = torch.sign(r_i)
        r_o = torch.sign(r_o + 0.1) # prevent zero r_o
        E = torch.mean(torch.abs(r_i),dim=(1,2,3),keepdim=True)
        E = torch.tanh(E)
        E = STE.apply(E, 2**e_bit)
        r_o = r_o * E
        return r_o

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clone(), None
   
# weight quantization in DoReFa-Net (Eq.(9), modified when k > 1)
class WeightQuantization(nn.Module):
    def __init__(self, w_bit: int, e_bit: int):
        super().__init__()
        self.w_bit = w_bit
        self.e_bit = e_bit
    
    def forward(self, x: torch.Tensor):
        if self.w_bit == 32:
            return x
        elif self.w_bit == 1:
            return binary_quantize.apply(x, self.e_bit)
        else:
            return quantize_k.apply(torch.tanh(x) / 2 / torch.max(torch.abs(torch.tanh(x))) + 1/2, self.w_bit) * (2**self.w_bit-1)/(2**(self.w_bit-1)) - 1
            
# activation quantization in DoReFa-Net (Eq.(11))
class ActivationQuantization(nn.Module):
    def __init__(self, a_bit: int):
        super().__init__()
        self.a_bit = a_bit
    
    def forward(self, x: torch.Tensor):
        if self.a_bit == 32:
            return x
        else:
            return quantize_k.apply(x, self.a_bit)
        
class ReLU8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        y = torch.clamp(x, 0, 8)
        if x.requires_grad:
            ctx.save_for_backward(x)
        return y
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x = ctx.saved_tensors[0]
        grad_inputs = torch.where(torch.bitwise_and(x >= 0, x <= 8), grad_outputs.clone(), torch.zeros_like(grad_outputs))
        return grad_inputs
 
def reshape_to_activation(input):
    return input.reshape(1, -1, 1, 1)
def reshape_to_weight(input):
    return input.reshape(-1, 1, 1, 1)
def reshape_to_bias(input):
    return input.reshape(-1) 

class quan_Conv2d_BNFold(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels, 
        kernel_size,
        a_bits=8,
        w_bits=8,
        e_bits=8,
        stride=1,
        padding=None,
        first_layer=False,
        relu=True,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-3,
        momentum=0.03
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.Tensor(out_channels))
        self.beta = nn.Parameter(torch.Tensor(out_channels))
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('first_bn', torch.zeros(1))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        torch.nn.init.kaiming_uniform_(self.weight.data)
        self.activation_quantization_function = ActivationQuantization(a_bits)
        self.weight_quantization_function = WeightQuantization(w_bits, e_bits)
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.e_bits = e_bits
        self.bias_quantization_function = WeightQuantization(e_bits, e_bits)
        self.first_layer = first_layer
        self.relu = relu
        self.layer_num_print = 0

    def forward(self, x):
        if self.first_layer:
            q_input = x
        else:
            q_input = self.activation_quantization_function(x)
        if self.training:
            weight_q = self.weight_quantization_function(self.weight)
            output = F.conv2d(
                input=q_input,
                weight=weight_q,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            dims = [dim for dim in range(4) if dim != 1]
            batch_mean = torch.mean(output, dim=dims)
            batch_var = torch.var(output, dim=dims)
            with torch.no_grad():
                if self.first_bn == 0:
                    self.first_bn.add_(1)
                    self.running_mean.add_(batch_mean)
                    self.running_var.add_(batch_var)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                    self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            if self.bias is not None:  
                bias = reshape_to_bias(self.beta + (self.bias -  batch_mean) * (self.gamma / torch.sqrt(batch_var + self.eps)))
            else:
                bias = reshape_to_bias(self.beta - batch_mean  * (self.gamma / torch.sqrt(batch_var + self.eps)))
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))     
        else:
            if self.bias is not None:
                bias = reshape_to_bias(self.beta + (self.bias - self.running_mean) * (self.gamma / torch.sqrt(self.running_var + self.eps)))
            else:
                bias = reshape_to_bias(self.beta - self.running_mean * (self.gamma / torch.sqrt(self.running_var + self.eps)))  # b融running
            weight = self.weight * reshape_to_weight(self.gamma / torch.sqrt(self.running_var + self.eps))  
        
        q_weight = self.weight_quantization_function(weight)
        if self.w_bits == 1:
            q_bias = self.bias_quantization_function(bias) / (2 * (2**self.e_bits - 1))
        else:
            q_bias = self.bias_quantization_function(bias) / (2**self.a_bits - 1)
        if self.training:  
          output = F.conv2d(
              input=q_input,
              weight=q_weight,
              bias=None,  
              stride=self.stride,
              padding=self.padding,
              dilation=self.dilation,
              groups=self.groups
          )
          
          output *= reshape_to_activation(torch.sqrt(self.running_var + self.eps) / torch.sqrt(batch_var + self.eps))
          output += reshape_to_activation(q_bias)

        else:
            if self.layer_num_print > 0:
                dir_path = os.path.join('./stats/', f'layer{self.layer_num_print}')
                if not os.path.exists(dir_path):  # 检查目录是否存在
                    os.makedirs(dir_path)  # 创建目录（支持多级目录）
                    print(f"目录 {dir_path} 已创建")
                with open(os.path.join('./stats/', f'layer{self.layer_num_print}.txt'), 'w') as f:
                    torch.set_printoptions(threshold=10000000000, linewidth=10000)

                    print(f'{self.a_bits}-bit Activation:', file=f)
                    print(q_input[0].shape, file=f)
                    print((q_input[0] * (2 ** self.a_bits - 1)).round().cpu().int(), file=f)

                    ori_input = (q_input[0] * (2 ** self.a_bits - 1)).round().cpu().int()
                    input_data, input_ddr_word_array = trans_conv_input_data(
                        os.path.join('./stats/', f'layer{self.layer_num_print}/', 'input_tensor.txt'),
                        ori_input,ori_input.shape[0], ori_input.shape[1], ori_input.shape[2])
                    
                    print(f'{self.w_bits}-bit Weight:', file=f)
                    if self.w_bits == 1:
                        E = torch.mean(torch.abs(weight),dim=(1,2,3),keepdim=True)
                        E = torch.tanh(E)
                        E = STE.apply(E, 2**self.e_bits)
                        print(E.shape, file=f)
                        print('E: ', (E * (2**self.e_bits)).flatten().cpu().int(), file=f)

                        ori_E = (E * (2**self.e_bits)).flatten().cpu().int()
                        E_data, E_ddr_words = trans_conv_E_data(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'E_tensor.txt'),
                                                                ori_E, 1, ori_E.shape[0])
                        generate_E_buf_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'E_buffer_init.txt'), E_ddr_words)

                        print(q_weight.shape, file=f)
                        print((q_weight / E).round().cpu().int(), file=f)

                        ori_weight = (q_weight / E).round().cpu().int()
                        ori_weight = ((1 - ori_weight) / 2).int() # 1 --> 0, -1 --> 1
                        weight_data, weights_ddr_words = trans_conv_weight_data(
                            os.path.join('./stats/', f'layer{self.layer_num_print}/', 'weights_tensor.txt'),
                            ori_weight, 1, ori_weight.shape[0], ori_weight.shape[1], ori_weight.shape[2])
                        
                    else:

                        E_data, E_ddr_words = trans_conv_E_data(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'E_tensor.txt'),
                                                                None, 0, q_weight.shape[0])
                        generate_E_buf_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'E_buffer_init.txt'), E_ddr_words)

                        print(q_weight.shape, file=f)
                        print((q_weight * (2 ** (self.w_bits - 1))).round().cpu().int(), file=f)

                        ori_weight = (q_weight * (2 ** (self.w_bits - 1))).round().cpu().int()
                        weight_data, weights_ddr_words = trans_conv_weight_data(
                            os.path.join('./stats/', f'layer{self.layer_num_print}/', 'weights_tensor.txt'),
                            ori_weight, 0, ori_weight.shape[0], ori_weight.shape[1], ori_weight.shape[2])
                        
                    
                    print(f'{self.bias_quantization_function.w_bit}-bit Bias:', file=f)
                    if self.w_bits == 1:
                        print(q_bias.shape, file=f)
                        print((q_bias * (2 * (2**self.e_bits - 1)) * (2**(self.e_bits-1))).round().cpu().int(), file=f)

                        ori_bias = (q_bias * (2 * (2**self.e_bits - 1)) * (2**(self.e_bits-1))).round().cpu().int()
                        bias_data, bias_ddr_words = trans_conv_bias_data(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'bias_tensor.txt'),
                                                                         ori_bias, 1, ori_bias.shape[0])
                        generate_bias_buf_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'bias_buffer_init.txt'), bias_ddr_words)
                    else:
                        print(q_bias.shape, file=f)
                        print((q_bias * (2**(self.w_bits-1)) * (2 ** self.a_bits - 1)).round().cpu().int(), file=f)
                        
                        ori_bias = (q_bias * (2**(self.w_bits-1)) * (2 ** self.a_bits - 1)).round().cpu().int()
                        bias_data, bias_ddr_words = trans_conv_bias_data(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'bias_tensor.txt'),
                                                                         ori_bias, 0, ori_bias.shape[0])
                        generate_bias_buf_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'bias_buffer_init.txt'), bias_ddr_words)
                    
                    if self.w_bits == 1:
                        # 8 * 256
                        print(q_bias.shape, file=f)
                        print(torch.ones(q_bias.shape, dtype=int) * 11, file=f)

                        ori_scale = torch.ones(q_bias.shape, dtype=int) * 11
                        scale_data, scale_ddr_words = trans_conv_scale_data(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'scale_tensor.txt'),
                                                                            ori_scale, 1, ori_scale.shape[0])
                        generate_scale_buf_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'scale_buffer_init.txt'),scale_ddr_words)
                    else:
                        # 8 * 128
                        print(q_bias.shape, file=f)
                        print(torch.ones(q_bias.shape, dtype=int) * 10, file=f)
                        
                        ori_scale = torch.ones(q_bias.shape, dtype=int) * 10
                        scale_data, scale_ddr_words = trans_conv_scale_data(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'scale_tensor.txt'),
                                                                            ori_scale, 0, ori_scale.shape[0])
                        generate_scale_buf_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'scale_buffer_init.txt'),scale_ddr_words)
                    
                    k,s,p,of,ix,iy,nif = \
                    ori_weight.shape[-1], self.stride[0], self.padding[0],self.out_channels, ori_input.shape[2],ori_input.shape[1],ori_input.shape[0]
                    ox = ix if s == 1 else ix // 2
                    oy = iy if s == 1 else iy // 2
                    input_base_adr = weights_ddr_words.shape[0] + bias_ddr_words.shape[0] + E_ddr_words.shape[0] + scale_ddr_words.shape[0]
                    if self.w_bits == 1:
                        generate_instr_args_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'instr_args_init.txt'),
                                                 os.path.join('./stats/', f'layer{self.layer_num_print}/', 'instr_args_hex_num_init.txt'),
                                                 1,k,s,p,of,ox,oy,ix,iy,nif,input_base_adr)
                        generate_ddr_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'DDR_init.txt'), 
                                          1, weights_ddr_words, bias_ddr_words, E_ddr_words, scale_ddr_words, input_ddr_word_array)
                    else:
                        generate_instr_args_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'instr_args_init.txt'),
                                                 os.path.join('./stats/', f'layer{self.layer_num_print}/', 'instr_args_hex_num_init.txt'),
                                                 0,k,s,p,of,ox,oy,ix,iy,nif,input_base_adr)
                        generate_ddr_init(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'DDR_init.txt'), 
                            0, weights_ddr_words, bias_ddr_words, E_ddr_words, scale_ddr_words, input_ddr_word_array)
                    
                # if self.layer_num_print > 2:
                #         exit()    
            output = F.conv2d(
                input=q_input,
                weight=q_weight,
                bias=q_bias,  
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        
        if self.relu:
            output = ReLU8.apply(output) / 8
            # output = F.relu(output)

            if not self.training and self.layer_num_print > 0:
                 with open(os.path.join('./stats/', f'layer{self.layer_num_print}.txt'), 'a') as f:
                    torch.set_printoptions(threshold=10000000000, linewidth=10000)
                    print(f'{self.a_bits}-bit Output:', file=f)
                    print(output.shape, file=f)
                    print((output * (2 ** self.a_bits - 1)).round().cpu().int(), file=f)

                    save_yolo_CBR(os.path.join('./stats/', f'layer{self.layer_num_print}/', 'output_tensor.txt'),
                                  (output * (2 ** self.a_bits - 1)).round().cpu().int())
        
        return output

class quan_Bottleneck(nn.Module):
    # Standard quantized bottleneck
    def __init__(self, c1, c2, shortcut=True, a_bits=8, w_bits=8, e_bits=8, groups=1, expansion=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * expansion)  # hidden channels
        self.add = shortcut and c1 == c2
        self.cv1 = quan_Conv2d_BNFold(c1, c_, kernel_size=1, stride=1, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits)
        self.cv2 = quan_Conv2d_BNFold(c_, c2, kernel_size=3, stride=1, groups=groups, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits, relu=(not self.add))

    def forward(self, x):
        if self.add:
            y = x + self.cv2(self.cv1(x))
            return ReLU8.apply(y) / 8
            #return F.relu(y)
        else:
            return self.cv2(self.cv1(x))

class quan_C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, a_bits=8, w_bits=8, e_bits=8, groups=1, expansion=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * expansion)  # hidden channels
        self.cv1 = quan_Conv2d_BNFold(c1, c_, kernel_size=1, stride=1, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits)
        self.cv2 = quan_Conv2d_BNFold(c1, c_, kernel_size=1, stride=1, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits)
        self.cv3 = quan_Conv2d_BNFold(2 * c_, c2, kernel_size=1, stride=1, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(quan_Bottleneck(c_, c_, shortcut=shortcut, groups=groups, expansion=1.0, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
      
class quan_SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, a_bits=8, w_bits=8, e_bits=8, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = quan_Conv2d_BNFold(c1, c_, kernel_size=1, stride=1, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits)
        self.cv2 = quan_Conv2d_BNFold(c_ * 4, c2, kernel_size=1, stride=1, a_bits=a_bits, w_bits=w_bits, e_bits=e_bits)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        # print("sppf")
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1)) 
          