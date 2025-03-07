import numpy as np
import math
import torch
import torch.nn.functional as F

# ddr arguments
ddr_word_width = 512

def int8_to_hex_complement(value):
    """
    将 int8 值转换为 2 位 16 进制补码形式
    """
    if value >= 0:
        return f"{value:02x}"  # 正数直接转换为 16 进制
    else:
        return f"{256 + value:02x}"  # 负数转换为补码


def generate_instr_args_init(args_file,args_hex_num_file,mode,k,s,p,of,ox,oy,ix,iy,nif,input_base_adr):
  pixels_in_row = 32
  buffers_num = 3
  pixels_in_row_real = 32
  buffers_num_real = 3
  
  # 定义变量
  mode_integer = mode
  k_real = k
  s_real = s
  p_real = p
  of_integer = of
  of_in_2pow_integer = int(math.log(of_integer, 2))
  ox_integer = ox
  ox_in_2pow_integer = int(math.log(ox_integer, 2))
  oy_integer = oy
  ix_integer = ix
  ix_in_2pow_integer = int(math.log(ix_integer, 2))
  iy_integer = iy
  nif_integer = nif
  if nif == 3:
     nif += 1
  nif_in_2pow_integer = int(math.log(nif, 2))
  row_num_real = 64 if mode_integer == 0 else 128  # 128

  # 计算表达式
  nif_mult_k_mult_k_integer = nif_integer * math.floor(k_real) * math.floor(k_real)  # nif * k * k
  N_chunks_integer = (
      math.ceil(ox_integer / pixels_in_row_real) *
      math.ceil(oy_integer / buffers_num_real) *
      math.ceil(of_integer / row_num_real)
  )
  bias_layer_base_buf_adr_rd_integer = 0
  E_layer_base_buf_adr_rd_integer = 0
  scale_layer_base_buf_adr_rd_integer = 0
  weights_layer_base_ddr_adr_rd_integer = 0
  input_ddr_layer_base_adr_integer = input_base_adr
  output_ddr_layer_base_adr_integer = 0
  ix_index_num_real = math.ceil(ix_integer / pixels_in_row_real)
  iy_index_num_real = math.ceil(iy_integer)
  tilex_first_ix_word_num_real = math.ceil(((pixels_in_row - 1) * s_real + k_real - p_real) / pixels_in_row)
  # tilex mid rectified
  tilex_last_ix_word_num_real = s_real if (((ix_index_num_real - tilex_first_ix_word_num_real) % s_real) == 0) \
  else ((ix_index_num_real - tilex_first_ix_word_num_real) % s_real)
  tilex_mid_ix_word_num_real = s_real
  tiley_first_iy_row_num_real = (buffers_num - 1) * s_real + k_real - p_real
  # tiley mid rectified
  tiley_last_iy_row_num_real = (buffers_num * s_real) if ((iy_index_num_real - tiley_first_iy_row_num_real) % (buffers_num * s_real) == 0) \
  else ((iy_index_num_real - tiley_first_iy_row_num_real) % (buffers_num * s_real))
  tiley_mid_iy_row_num_real = buffers_num * s_real
  of_div_row_num_ceil_real = math.ceil(of_integer / row_num_real)
  tiley_first_tilex_first_split_size_real = math.ceil(tiley_first_iy_row_num_real * tilex_first_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_first_tilex_last_split_size_real = math.ceil(tiley_first_iy_row_num_real * tilex_last_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_first_tilex_mid_split_size_real = math.ceil(tiley_first_iy_row_num_real * tilex_mid_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_last_tilex_first_split_size_real = math.ceil(tiley_last_iy_row_num_real * tilex_first_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_last_tilex_last_split_size_real = math.ceil(tiley_last_iy_row_num_real * tilex_last_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_last_tilex_mid_split_size_real = math.ceil(tiley_last_iy_row_num_real * tilex_mid_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_mid_tilex_first_split_size_real = math.ceil(tiley_mid_iy_row_num_real * tilex_first_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_mid_tilex_last_split_size_real = math.ceil(tiley_mid_iy_row_num_real * tilex_last_ix_word_num_real / of_div_row_num_ceil_real)
  tiley_mid_tilex_mid_split_size_real = math.ceil(tiley_mid_iy_row_num_real * tilex_mid_ix_word_num_real / of_div_row_num_ceil_real)
  instr_args_init_mem = [
      mode_integer,
      k_real,
      s_real,
      p_real,
      of_integer,
      of_in_2pow_integer,
      ox_integer,
      ox_in_2pow_integer,
      oy_integer,
      ix_integer,
      ix_in_2pow_integer,
      iy_integer,
      nif_integer,
      nif_in_2pow_integer,
      nif_mult_k_mult_k_integer,
      N_chunks_integer,
      E_layer_base_buf_adr_rd_integer,
      bias_layer_base_buf_adr_rd_integer,
      scale_layer_base_buf_adr_rd_integer,
      weights_layer_base_ddr_adr_rd_integer,
      input_ddr_layer_base_adr_integer,
      output_ddr_layer_base_adr_integer,
      tilex_first_ix_word_num_real,
      tilex_last_ix_word_num_real,
      tilex_mid_ix_word_num_real,
      tiley_first_iy_row_num_real,
      tiley_last_iy_row_num_real,
      tiley_mid_iy_row_num_real,
      ix_index_num_real,
      iy_index_num_real,
      of_div_row_num_ceil_real,
      tiley_first_tilex_first_split_size_real,
      tiley_first_tilex_last_split_size_real,
      tiley_first_tilex_mid_split_size_real,
      tiley_last_tilex_first_split_size_real,
      tiley_last_tilex_last_split_size_real,
      tiley_last_tilex_mid_split_size_real,
      tiley_mid_tilex_first_split_size_real,
      tiley_mid_tilex_last_split_size_real,
      tiley_mid_tilex_mid_split_size_real
  ]
  # 将 instr_args_init_mem 中的每个元素写入到 instr_args_init.txt 文件中
  with open(args_file, "w") as file:
      for value in instr_args_init_mem:
          hex_value = format(value, '08X')  # 转换为 8 位 16 进制数，不足 8 位时前面补零
          file.write(f"{hex_value}\n")

  instr_args_bit_lengths = [
      4,
      4,
      4,
      4,
      16,
      4,
      16,
      4,
      16,
      16,
      4,
      16,
      16,
      4,
      32,
      32,
      16,
      16,
      16,
      32,
      32,
      32,
      8,
      8,
      8,
      8,
      8,
      8,
      16,
      16,
      8,
      8,
      8,
      8,
      8,
      8,
      8,
      8,
      8,
      8
  ]
  with open(args_hex_num_file, "w") as file:
    hex_value = combine_parameters_to_hex(instr_args_init_mem, instr_args_bit_lengths)
    file.write(f"{hex_value}\n")

def combine_parameters_to_hex(params, bit_lengths):
    """
    将参数组合成一个512位的二进制数并转换为16进制。

    :param params: 参数列表，包含38个参数的值
    :param bit_lengths: 每个参数对应的二进制位数列表
    :return: 512位二进制数的16进制表示
    """
    if len(params) != len(bit_lengths):
        raise ValueError("参数数量和位数列表长度不匹配！")
    
    result = 0  # 初始化结果为0
    offset = 0  # 位偏移量

    for param, bit_length in zip(params, bit_lengths):
        # 检查参数是否超出指定的位数范围
        if param >= (1 << bit_length):
            raise ValueError(f"参数值 {param} 超出了 {bit_length} 位的范围！")
        
        # 将参数值左移至正确的位置并拼接到结果中
        result |= param << offset
        offset += bit_length  # 更新偏移量

    # 确保结果为512位
    if offset > 512:
        raise ValueError("参数总位数超过了512位！")
    
    # 将结果转换为16进制并返回
    hex_result = hex(result)[2:].upper().zfill(128)  # 512位对应128个16进制字符
    return hex_result

def trans_conv_weight_data(weights_tensor_file, ori_weights, mode, of, nif, k):
  # trans ori_weights into DDR format
  # ori_weights[of, nif, k, k]
  weight_data = ori_weights
  img2col_weight_data = weight_data.reshape(of, nif*k*k)
  ## reshape weight data
  # the amount of out channel of weights per complete ddr word
  weight_width = 8 if mode == 0 else 1
  out_channel_parallel = 64 if mode == 0 else 128
  weights_out_channel_in_word = min(ddr_word_width // weight_width, out_channel_parallel, of) #64 or 128 or F
  # weights_tensor[FP, ID*K*K]
  weights_tensors = img2col_weight_data.split(weights_out_channel_in_word, dim=0)
  weights_tensor_list = []
  print("Weights shape (Output Channels, Input Channels, Kernel Height, Kernel Width):", weight_data.shape)
  print(f"weight tensors shape:")
  for index, weights_tensor in enumerate(weights_tensors):
      print(f"index: {index} \n weights tensor shape: {weights_tensor.shape} \n {weights_tensor} \n")
      weights_data_in_word = ddr_word_width // weight_width
      kernel_num = weights_tensor.shape[0] #output channels indeed
      # each weight word contains at most 64 weights from 64 kernels
      # permuted weights_tensor[k*k*ID, FP], every row is a ddr word carrying FP weights
      weights_tensor_list.extend(
          [F.pad(wt, (0, weights_data_in_word - kernel_num)) 
          for wt in weights_tensor.permute(1,0).reshape(-1,).split(kernel_num, dim=0)]
          )
  weights_ddr_words = torch.stack(weights_tensor_list, dim=0)
  print(f"weights ddr words: {weights_ddr_words}")
  ## write weight into weight txt
  with open(weights_tensor_file, "w") as f:
      for weight_ddr_word_index, weight_ddr_word in enumerate(weights_ddr_words):
      # iterate each weight word
          dec_str = ''.join([f'{weight_num:03d} ' for weight_num in weight_ddr_word])
          f.write(dec_str + '\n')
  return weight_data, weights_ddr_words

def trans_conv_input_data(input_tensor_file, ori_input, nif, iy, ix):
  # generate and return conv input, then split and return it 
  # input[ID, IH, IW]
  # input channel num should be an even num. 
  # if not, expand 3 channels -> 4 channels, last channel is 0
  # uint8 [0,255]
  # if nif is odd, add a zero channel to make it even
  if nif % 2 != 0:
    input_data = ori_input
    zero_channel = torch.zeros((1, iy, ix), dtype=torch.int)
    input_data = torch.cat((input_data, zero_channel), dim=0)
    # nif += 1
  else:
    input_data = ori_input 

  # reshape input tensor into ddr words
  activation_x_num_in_ddr_word = 32
  activation_in_channel_num_in_ddr_word = 2 # ddr_word_width / activation_x_num_in_ddr_word / weight_word_width
  assert ix % 32 == 0
  # reshape input data
  # (C, H, W) ---> (H, W/32, C/2, 32*2); input ddr words in tensor format
  input_tensor = input_data.view(math.ceil(nif / 2), 2, iy, ix // 32, 32)\
      .permute(2, 3, 0, 1, 4).contiguous().view(iy, ix // 32, math.ceil(nif / 2), 64)
  print("Input data shape (Input Channels, Input Height, Input Width):", input_data.shape)
  print(f"input tensors shape: {input_tensor.shape}")
  print(input_tensor)
  input_tensor_shape = input_tensor.shape
  input_ddr_word_array = input_tensor.reshape(-1).split(split_size=input_tensor_shape[-1], dim=0)    
  ## write input into input txt
  with open(input_tensor_file, "w") as f:
      for input_ddr_word_index, input_ddr_word in enumerate(input_ddr_word_array):
      # iterate each input word
          dec_str = ''.join([f'{input_num:03d} ' for input_num in input_ddr_word])
          f.write(dec_str + '\n')
  return input_data, input_ddr_word_array
 
def trans_conv_bias_data(bias_tensor_file, ori_bias, mode, of):
  # ori_bias[F]
  # int8 [-128,127]
  bias_data = ori_bias.reshape(1, of)
  ## reshape bias data
  # the amount of out channel of bias per complete ddr word
  bias_width = 8
  out_channel_parallel = 64 if mode == 0 else 128 
  bias_out_channel_in_word = min(ddr_word_width // bias_width, out_channel_parallel, of) #64 or F
  bias_tensors = bias_data.split(bias_out_channel_in_word, dim=1)
  bias_tensor_list = [] #bias ddr words in tensor format
  print(f"bias tensors shape:")
  for index, bias_tensor in enumerate(bias_tensors):
      print(f"index: {index} \n bias tensor shape: {bias_tensor.shape} \n {bias_tensor} \n")
      bias_data_in_word = ddr_word_width // bias_width
      bias_num = bias_tensor.shape[1]
      # each bias word contains 64 bias from 64 kernels
      # bias_tensor[1,FP] is a ddr word carrying 64 bias
      bias_tensor_list.append(F.pad(bias_tensor.reshape(-1,), (0, bias_data_in_word - bias_num)))
  bias_ddr_words = torch.stack(bias_tensor_list, dim=0)
  print(f"bias ddr words: {bias_ddr_words}")
  ## write bias into bias txt
  with open(bias_tensor_file, "w") as f:    
      for bias_ddr_word_index, bias_ddr_word in enumerate(bias_ddr_words):
      # iterate each bias word
          dec_str = ''.join([f'{bias_num:03d} ' for bias_num in bias_ddr_word])
          f.write(dec_str + '\n')
  return bias_data, bias_ddr_words

def trans_conv_E_data(E_tensor_file, ori_E_data, mode, of):
  # E[F]
  # uint16 [0,256*256-1]
  # 1
  E_data = (torch.ones(of, dtype=torch.int) * (torch.tensor([1], dtype=int))) \
    if (mode == 0) else ori_E_data
  ## reshape e_scale_tail data
  # the amount of out channel of tail per complete ddr word 
  E_width = 16
  out_channel_parallel = 64 if mode == 0 else 128 
  E_out_channel_in_word = min(ddr_word_width // E_width, out_channel_parallel, of) #32 or F
  E_tensors = E_data.split(E_out_channel_in_word, dim=0)
  E_tensor_list = [] #tail ddr words in tensor format
  print(f"E tensors shape:")
  for index, E_tensor in enumerate(E_tensors):
      print(f"index: {index} \n E tensor shape: {E_tensor.shape} \n {E_tensor} \n")
      E_data_in_word = ddr_word_width // E_width
      E_num = E_tensor.shape[0]
      # each E word contains 64 E from 64 kernels
      # E_tensor[FP] is a ddr word carrying 64 E
      E_tensor_list.append(F.pad(E_tensor, (0, E_data_in_word - E_num)))
  E_ddr_words = torch.stack(E_tensor_list, dim=0)
  print(f"E ddr words: {E_ddr_words}")
  ## write E into E txt
  with open(E_tensor_file, "w") as f:
      for E_ddr_word_index, E_ddr_word in enumerate(E_ddr_words):
      # iterate each E word
          dec_str = ''.join([f'{E_num:06d} ' for E_num in E_ddr_word])
          f.write(dec_str + '\n')
  return E_data, E_ddr_words

def trans_conv_scale_data(scale_tensor_file, ori_scale_data, mode, of):
  # scale[F]
  # uint8 [0,256] scale_scalar
  scale_data = ori_scale_data
  ## reshape scale data
  # the amount of out channel of scale per complete ddr word 
  scale_width = 8
  out_channel_parallel = 64 if mode == 0 else 128 
  scale_out_channel_in_word = min(ddr_word_width // scale_width, out_channel_parallel, of) #64 or F
  scale_tensors = scale_data.split(scale_out_channel_in_word, dim=0)
  scale_tensor_list = [] #scale ddr words in tensor format
  print(f"scale tensors shape:")
  for index, scale_tensor in enumerate(scale_tensors):
      print(f"index: {index} \n scale tensor shape: {scale_tensor.shape} \n {scale_tensor} \n")
      scale_data_in_word = ddr_word_width // scale_width
      scale_num = scale_tensor.shape[0]
      # each scale word contains 64 scale from 64 kernels
      # scale_tensor[FP] is a ddr word carrying 64 scale
      scale_tensor_list.append(F.pad(scale_tensor, (0, scale_data_in_word - scale_num)))
  scale_ddr_words = torch.stack(scale_tensor_list, dim=0)
  print(f"scale ddr words: {scale_ddr_words}")
  ## write scale into scale txt
  with open(scale_tensor_file, "w") as f:
      for scale_ddr_word_index, scale_ddr_word in enumerate(scale_ddr_words):
      # iterate each scale word
          dec_str = ''.join([f'{scale_num:03d} ' for scale_num in scale_ddr_word])
          f.write(dec_str + '\n')
  return scale_data, scale_ddr_words

def generate_ddr_init(ddr_init_file, mode, weights_ddr_words, bias_ddr_words, E_ddr_words, scale_ddr_words, input_ddr_word_array):
  ## write num in complementary code into txt file.
  ## write DDR words in txt file 
  with open(ddr_init_file, "w") as f:
      # weight
      weights_ddr_words_shape = weights_ddr_words.shape
      #turn 2d tensor into 1d tensor array
      weights_ddr_word_array = weights_ddr_words.reshape(-1,).split(split_size=weights_ddr_words_shape[-1], dim=0)
      for weight_ddr_word_index, weight_ddr_word in enumerate(weights_ddr_word_array):
      # iterate each weight word
          reverse_weight_ddr_word = torch.flip(weight_ddr_word, dims=[0])
          if mode == 0:
              hex_str = ''.join([int8_to_hex_complement(weight_num.item()) for weight_num in reverse_weight_ddr_word])
              # print(hex_str + ',\n')
              f.write(hex_str + '\n')
          else:
              hex_array = []
              reverse_weight_ddr_word_length = reverse_weight_ddr_word.shape[-1]
              assert reverse_weight_ddr_word_length == 512
              for weight_num_index in range(0, reverse_weight_ddr_word_length, 4):
                  # 0,1,2,3,... --> high of index, low of index
                  weight_bit1 = reverse_weight_ddr_word[weight_num_index]
                  weight_bit2 = reverse_weight_ddr_word[weight_num_index+1]
                  weight_bit3 = reverse_weight_ddr_word[weight_num_index+2]
                  weight_bit4 = reverse_weight_ddr_word[weight_num_index+3]
                  weight_str = ''.join([f'{weight_bit1:01b}', f'{weight_bit2:01b}',  f'{weight_bit3:01b}',  f'{weight_bit4:01b}'])
                  weight = int(weight_str, base=2)
                  hex_array.append(f'{weight:01x}')
              hex_str = ''.join(hex_array)
              f.write(hex_str + '\n')
      # f.write('\n')   
      
      # bias
      bias_ddr_words_shape = bias_ddr_words.shape
      bias_ddr_word_array = bias_ddr_words.reshape(-1,).split(split_size=bias_ddr_words_shape[-1], dim=0)
      for bias_ddr_word_index, bias_ddr_word in enumerate(bias_ddr_word_array):
      # iterate each bias word
          reverse_bias_ddr_word = torch.flip(bias_ddr_word, dims=[0])
          hex_str = ''.join([int8_to_hex_complement(bias_num.item()) for bias_num in reverse_bias_ddr_word])
          # print(hex_str + ',\n')
          f.write(hex_str + '\n')
      # f.write('\n')
      
      # E
      E_ddr_words_shape = E_ddr_words.shape
      E_ddr_word_array = E_ddr_words.reshape(-1,).split(split_size=E_ddr_words_shape[-1], dim=0)
      for E_ddr_word_index, E_ddr_word in enumerate(E_ddr_word_array):
      # iterate each E word
          for num_index in range(E_ddr_words_shape[-1]-1, -1, -1):
              E_num = E_ddr_word[num_index]
              hex_str = ''.join(f'{E_num:04x}')
              # print(hex_str + ',\n')
              f.write(hex_str)
          f.write('\n')
      # f.write('\n')
      
      # scale
      scale_ddr_words_shape = scale_ddr_words.shape
      scale_ddr_word_array = scale_ddr_words.reshape(-1,).split(split_size=scale_ddr_words_shape[-1], dim=0)
      for scale_ddr_word_index, scale_ddr_word in enumerate(scale_ddr_word_array):
      # iterate each scale word
          reverse_scale_ddr_word = torch.flip(scale_ddr_word, dims=[0])
          hex_str = ''.join([f'{scale_num:02x}' for scale_num in reverse_scale_ddr_word])
          # print(hex_str + ',\n')
          f.write(hex_str + '\n')
      # f.write('\n')    
      
      # input
      input_ddr_word_len = len(input_ddr_word_array)
      # iterate each input word
      for input_ddr_word_index, input_ddr_word in enumerate(input_ddr_word_array):
          reverse_input_ddr_word = torch.flip(input_ddr_word, dims=[0])
          # concate each num in a word to a hex num
          hex_str = ''.join([int8_to_hex_complement(input_num.item()) for input_num in reverse_input_ddr_word])
          # print(f"{hex_str} \n")
          f.write(hex_str)
          if input_ddr_word_index < input_ddr_word_len - 1:
              f.write('\n')

def generate_bias_buf_init(bias_init_file, bias_ddr_words):
  with open(bias_init_file, "w") as f:
      # bias
      bias_ddr_words_shape = bias_ddr_words.shape
      bias_ddr_word_array = bias_ddr_words.reshape(-1,).split(split_size=bias_ddr_words_shape[-1], dim=0)
      bias_ddr_word_len = len(bias_ddr_word_array)
      for bias_ddr_word_index, bias_ddr_word in enumerate(bias_ddr_word_array):
      # iterate each bias word
          reverse_bias_ddr_word = torch.flip(bias_ddr_word, dims=[0])
          hex_str = ''.join([int8_to_hex_complement(bias_num.item()) for bias_num in reverse_bias_ddr_word])
          # print(hex_str + ',\n')
          f.write(hex_str)
          if bias_ddr_word_index < bias_ddr_word_len - 1:
              f.write('\n')

def generate_E_buf_init(E_init_file,E_ddr_words):
  with open(E_init_file, "w") as f:
      # E
      E_ddr_words_shape = E_ddr_words.shape
      E_ddr_word_array = E_ddr_words.reshape(-1,).split(split_size=E_ddr_words_shape[-1], dim=0)
      E_ddr_word_len = len(E_ddr_word_array)
      for E_ddr_word_index, E_ddr_word in enumerate(E_ddr_word_array):
      # iterate each E word
          for num_index in range(E_ddr_words_shape[-1]-1, -1, -1):
              E_num = E_ddr_word[num_index]
              hex_str = ''.join(f'{E_num:04x}')
              # print(hex_str + ',\n')
              f.write(hex_str)
          if E_ddr_word_index < E_ddr_word_len - 1:
              f.write('\n')

def generate_scale_buf_init(scale_init_file, scale_ddr_words):
  with open(scale_init_file, "w") as f:
      # scale
      scale_ddr_words_shape = scale_ddr_words.shape
      scale_ddr_word_array = scale_ddr_words.reshape(-1,).split(split_size=scale_ddr_words_shape[-1], dim=0)
      scale_ddr_word_len = len(scale_ddr_word_array)
      for scale_ddr_word_index, scale_ddr_word in enumerate(scale_ddr_word_array):
      # iterate each scale word
          reverse_scale_ddr_word = torch.flip(scale_ddr_word, dims=[0])
          hex_str = ''.join([f'{scale_num:02x}' for scale_num in reverse_scale_ddr_word])
          # print(hex_str + ',\n')
          f.write(hex_str)
          if scale_ddr_word_index < scale_ddr_word_len - 1:
              f.write('\n')

def save_yolo_CBR(output_file, output_tensor):
    with open(output_file, "w") as f: 
      # 第一行写入维度信息
      f.write(" ".join(map(str, output_tensor.shape)) + "\n")
      # 遍历张量的每个元素并写入文件
      for i in range(output_tensor.size(0)):  # 遍历 batch 维度
          for j in range(output_tensor.size(1)):  # 遍历通道维度
              for k in range(output_tensor.size(2)):  # 遍历高度维度
                  f.write(f"channel {j:4d} - height {k:4d} : ")
                  for value in output_tensor[i, j, k, :]:  # 遍历宽度维度
                      f.write(f"{int(value.item()):3d} ")  # 写入整数值
                  f.write("\n")  # 每一行结束后换行