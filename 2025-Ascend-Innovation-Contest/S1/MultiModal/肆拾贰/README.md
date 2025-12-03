# MindNLP 模型优化 (Qwen2-VL & janus_pro)

本文档详细记录了针对 Qwen2-VL 和 janus_pro 模型的关键性能优化点，并附带了相应的核心代码实现。



## 一、Qwen2-VL 模型优化

### 1、使用融合算子

#### ① RoPE：mindspore.ops.rotary_position_embedding

修改前：

```python
mrope_section = mrope_section * 2
cos = ops.cat([m[i % 3] for i, m in enumerate(ops.split(cos, mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
sin = ops.cat([m[i % 3] for i, m in enumerate(ops.split(sin, mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

修改后：

```python
q_embed = mindspore.ops.rotary_position_embedding(q, cos, sin)
k_embed = mindspore.ops.rotary_position_embedding(k, cos, sin)
```



#### ② RMSNorm：mindnlp.core.nn.rms_norm

修改前：

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(mindspore.float32)
variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

修改后：

```python
return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```



#### ③ FlashAttention

- 在 VisionAttention 中，使用 mindspore.ops.flash_attention_score，需要对 qk 先进行 scale，$\frac{q}{\sqrt{\sqrt{d}}}$， $\frac{k}{\sqrt{\sqrt{d}}}$，然后计算 flash_attention 时 scale 设为默认 1.0，否则精度不对齐（感觉可能跟大算子底层的计算顺序有关系，但这个方法只在这里有用，迁到 janus_pro 模型还是 mismatch）

```python
self.scalar_value = 1 / math.sqrt(math.sqrt(self.head_dim))
seq_length = hidden_states.shape[0]
q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb) * self.scalar_value
k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb) * self.scalar_value
attn_output = mindspore.ops.flash_attention_score(q, k, v.unsqueeze(0), self.num_heads, input_layout='BSND')
attn_output = attn_output.reshape(seq_length, -1)
attn_output = self.proj(attn_output)
```



- 在 Qwen2VLAttention 中，prefill 阶段，使用 mindspore.ops.fused_infer_attention_score，decoder阶段保持原来的计算，全部使用 flash_attention 会导致精度不对齐

```python
if query_states.shape[-2] != 1:  # 判定 prefill 阶段还是 decoder 阶段
    attn_mask = (attention_mask != 0).to(dtype=mindspore.uint8)
    attn_output = mindspore.ops.fused_infer_attention_score(query_states*self.scalar_value, key_states*self.scalar_value, value_states, num_key_value_heads=self.num_key_value_heads, num_heads=self.num_heads, input_layout='BNSD', atten_mask=attn_mask)[0]

else:
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = ops.matmul(query_states, mint.permute(key_states, (0, 1, 3, 2))) / self.head_dim_sqrt
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=mindspore.bfloat16)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = ops.matmul(attn_weights, value_states)
```



### 2、mint 算子替换

#### ① nn.Conv3d 改用 mindspore.mint.Conv3D，需要进行权重转换

修改前：
```python
self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
```

修改后：

```python
self.proj = mint.nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False, dtype=mindspore.bfloat16)
```



#### ② .swapaxes 改用 mindspore.mint.permute



### 3、旋转位置编码优化

预计算 sin / cos 表，避免在前向传播中重复计算



### 4、其它改进

#### ① Qwen2VLAttention 的 q_proj、k_proj、v_proj 合成一个 w_qkv

修改前：

```python
def __intit__():
	self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

def forward():
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
```

修改后：

```python
def __intit__():
    self.w_qkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim + self.num_key_value_heads * self.head_dim * 2, bias=True)

def forward():
    qkv = self.w_qkv(hidden_states)
    query_states, key_states, value_states = ops.split(qkv, [self.hidden_size,  self.num_key_value_heads * self.head_dim,  self.num_key_value_heads * self.head_dim], dim=2)
```



#### ② repeat_kv 优化

修改前：

```python
def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

修改后：

```python
def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    return ops.repeat_interleave(hidden_states, repeats=n_rep, dim=1)
```



## 二、janus_pro 模型优化

### 1、数据预处理（主要的性能瓶颈所在）

#### ① 重写 VLChatProcessor 的处理逻辑

原始的方法中存在 `image_token_mask = input_ids == self.image_id` 以及 `batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id` 等使用 `==` 逐元素比较的方法，很慢，参考 qwen2-vl 的方法去生成 input_ids，以及重写 images_seq_mask 的生成逻辑，避免使用 `==`

```python
class VLChatProcessor(ProcessorMixin):
    def process_one():
		# 此处只给出核心改进代码
        tmp_sft_format = sft_format
        tmp_sft_format = tmp_sft_format.split(self.image_tag)[0]
        tmp_input_ids = self.tokenizer.encode(tmp_sft_format)
        tmp_mask_before_len = len(tmp_input_ids)
        mask = [0] * tmp_mask_before_len

        index = 0
        while self.image_tag in sft_format:
            mask += [0]
            sft_format = sft_format.replace(
                self.image_tag, self.image_start_tag+"<|placeholder|>"*self.num_image_tokens+self.image_end_tag, 1
            )
            mask += [1] * self.num_image_tokens
            index += 1
        sft_format = sft_format.replace("<|placeholder|>", self.image_tag)
        num_image_tokens = mindspore.Tensor([self.num_image_tokens] * index, mindspore.int32)

        # tokenize
        input_ids = self.tokenizer.encode(sft_format)
        tmp_mask_last_len = len(input_ids) - len(mask)
        mask += [0] * tmp_mask_last_len
        images_seq_mask = mindspore.Tensor(mask, dtype=mindspore.bool_)
        input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
		
        # ...
        return prepare, images_seq_mask
```



#### ② 使用 opencv 代替 PIL 加载图像

opencv 读取图像的速度大概是 PIL 的10倍左右，但这块对整体的提升不大，主要瓶颈在 resize、rescale 等操作上。

前期尝试过使用 opencv 加载图像后，用 numpy 重写数据预处理过程，但是遇到 ms.dataset.vision.Resize 的 BICUBIC 插值对针对相同数据但不同格式（PIL 和 numpy）存在精度误差，导致最终 mismatch，没找到好的解决方法。



### 2、其它改进（与 Qwen2-VL 模型类似）

#### ① 使用融合算子 F.rms_norm

#### ② 旋转位置编码优化——预计算 sin / cos 表，避免在前向传播中重复计算

#### ③ repeat_kv 优化

#### ④ rotate_half 优化

修改前：

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), dim=-1)
```

修改后：

```python
def rotate_half(x):
    x1, x2 = ops.split(x, x.shape[-1] // 2, dim=-1)
    return ops.cat((-x2, x1), dim=-1)
```

#### 

## 三、最终收益

| model_name | memory_reserved | memory_allocated | avg_prefill_latency | avg_decode_latency  |
| ---------- | --------------- | ---------------- | ------------------- | ------------------- |
| Qwen2-VL   | 6.442450944     | 5.672920576      | 0.2023613452911377  | 0.04043297529220581 |
| janus_pro  | 17.179869184    | 15.238398464     | 0.13930201530456543 | 0.04886315107345581 |



## 四、评测结果

| 评测指标       | 平均得分     |
| -------------- | ------------ |
| 峰值显存得分   | 116.6667     |
| Prefill时延    | 425.6324     |
| Decode时延得分 | 208.4923     |
| **总分**       | **250.2638** |