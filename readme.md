# 项目使用流程说明

本项目主要分为两个阶段：**SSQR-Stage1** 和 **SSQR-Stage2**。以下为详细步骤说明。

---

## 🔹 SSQR-Stage1

### 第一步：运行 `run.py`
- **作用**：得到 `ent_code` 和 `ent_code_emb`。
- **需要配置的参数**：
  - `--data`

---

### 第二步：运行 `train_linking_predict.py`
- **作用**：训练一个传统的图神经网络用于进行链路预测任务。
- **需要配置的参数**：
  - `--dataset`  
- **输出结果**：得到一个模型，用于后续链路预测任务，在指令微调模板中提供候选实体。

---

### 第三步：运行 `preprocess.py`
- **作用**：生成“候选 data_file” (`.pt`) 文件。
- **输出文件内容**：

```json
{
  "eneity_list": List[str],
  "relation_list": List[str],
  "valid_pre": List[List[int]],  // 每条: [h, r, t, cand1, cand2, ...]
  "test_pre":  List[List[int]]
}
```

- **需要配置的参数**：
  - `DATA_ROOT`：数据集文件夹路径  
  - `OUT_FILE`：数据文件路径  
  - `MODEL_TYPE`：提供候选的模型类型  
  - `CHECKPOINT`：模型地址（第二步训练得到的模型）  

---

### 第四步：运行 `gen_adaprop.py`
- **作用**：生成指令微调需要的训练模板文件。
- **需要配置的参数**：
  - `data_file`：上一步得到的文件  
  - `code_file`：第一步得到的实体编码文件  
  - `data_dir`：数据集所在文件夹路径  

---

## 🔹 SSQR-Stage2

### 第一步：运行 `script/add_token.py`
- **作用**：将量化码扩充进词表。
- **需要配置的参数**：
  - `--base_model`：本地 LLaMA 模型位置  

---

### 第二步：运行 `script/load_data.py`
- **作用**：将微调模板转为 **LLaMA Factory** 可处理的格式。  
  - 得到调整好的数据集后，需要在 `llama factory/data/dataset_info.json` 中配置本地数据集。  
  - **数据调整规则**：  
    1. 如果 `completion` 中已有答案 → 保持不变。  
    2. 如果没有答案 → 将正确答案加入并放在排序第一位，其余依次后移，取前 `topk`。  

- **需要配置的参数**：
  - `--in_dir`：前面得到的原始微调指令模板文件夹位置  
  - `--out_dir`：保存地址  

---

### 第三步：进行指令微调
- **运行脚本**：  
  ```bash
  project/SSQR-stage2/LLaMA-Factory/set_up.sh
  ```
- **说明**：脚本内的参数均可自行调整。

---

## 📌 总结
1. **Stage1**：构建实体编码、链路预测模型、候选生成与训练模板。  
2. **Stage2**：扩充词表、格式转换、指令微调。  

完整流程执行后，即可得到最终可用于下游任务的微调模型。
