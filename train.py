import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import logging
from datetime import datetime
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
model_path = "yourself model path"


# 定义数据集类
class MedicalDataset:
    def __init__(self, texts, labels, tokenizer, max_length=128, batch_size=16, shuffle=True):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(labels)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.indices = list(range(self.num_samples))
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        batch_texts = [self.texts[i] for i in batch_indices]
        batch_labels = self.labels[batch_indices]

        # 对文本进行编码
        encodings = self.tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        self.current_batch += 1

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': batch_labels
        }

    def __len__(self):
        return self.num_batches


# 定义BERT模型
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)  # 增加dropout
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, disease_map=None, num_epochs=5,
                early_stopping_patience=3):
    best_val_loss = float('inf')
    best_val_f1 = 0
    early_stopping_counter = 0

    # 记录每个epoch的训练和验证指标
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rates': []
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        batch_count = 0
        all_train_preds = []
        all_train_labels = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            batch_count += 1

            # 每100个batch打印一次详细信息
            if batch_count % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                preds = torch.argmax(outputs, dim=1)
                batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
                print(f"\nBatch {batch_count}:")
                print(f"Loss: {loss.item():.4f}")
                print(f"Batch Accuracy: {batch_acc:.4f}")
                print(f"Learning Rate: {current_lr:.2e}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            # 收集训练预测结果
            preds = torch.argmax(outputs, dim=1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        # 验证
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # 计算训练和验证指标
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)

        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)

        # 打印详细的epoch信息
        logging.info(f'\nEpoch {epoch + 1} 详细报告:')
        logging.info(f'训练集 - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}')
        logging.info(f'验证集 - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}')
        logging.info(f'当前学习率: {current_lr:.2e}')

        # 打印每个类别的准确率
        logging.info('\n各类别准确率:')
        for i, disease in enumerate(disease_map.keys()):
            train_mask = np.array(all_train_labels) == i
            val_mask = np.array(all_val_labels) == i
            if np.any(train_mask) and np.any(val_mask):
                train_acc = accuracy_score(np.array(all_train_labels)[train_mask],
                                           np.array(all_train_preds)[train_mask])
                val_acc = accuracy_score(np.array(all_val_labels)[val_mask],
                                         np.array(all_val_preds)[val_mask])
                logging.info(f'{disease}: 训练集 {train_acc:.4f}, 验证集 {val_acc:.4f}')

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_f1': val_f1,
                'disease_map': disease_map,
                'num_classes': len(disease_map),
                'history': history
            }, 'best_bert_model.pth')
        else:
            early_stopping_counter += 1

        # 早停
        if early_stopping_counter >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break


def evaluate_model(model, test_loader, criterion, disease_map):
    model.eval()
    total_test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    logging.info("\n测试集评估结果：")
    logging.info(f'Average test loss: {avg_test_loss:.4f}')
    logging.info(f'Test Accuracy: {test_accuracy:.4f}')
    logging.info(f'Test F1 Score: {test_f1:.4f}')
    logging.info("\n分类报告：")
    logging.info(classification_report(all_labels, all_preds,
                                       target_names=list(disease_map.keys())))


def main():
    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)

    # 加载数据
    train_data = load_json_data('data/train.json')
    val_data = load_json_data('data/val.json')
    test_data = load_json_data('data/test.json')

    # 提取文本和标签
    train_texts = train_data["texts"]
    train_labels = train_data["labels"]
    val_texts = val_data["texts"]
    val_labels = val_data["labels"]
    test_texts = test_data["texts"]
    test_labels = test_data["labels"]

    # 打印原始标签示例
    logging.info("原始训练集标签示例：")
    logging.info(train_labels[:5])  # 显示前5个标签
    logging.info("原始验证集标签示例：")
    logging.info(val_labels[:5])
    logging.info("原始测试集标签示例：")
    logging.info(test_labels[:5])

    # 检查数据集大小
    logging.info(f"训练集大小: {len(train_texts)}")
    logging.info(f"验证集大小: {len(val_texts)}")
    logging.info(f"测试集大小: {len(test_texts)}")

    # 检查是否有重复数据
    train_set = set(zip(train_texts, train_labels))
    val_set = set(zip(val_texts, val_labels))
    test_set = set(zip(test_texts, test_labels))

    train_val_overlap = len(train_set.intersection(val_set))
    train_test_overlap = len(train_set.intersection(test_set))
    val_test_overlap = len(val_set.intersection(test_set))

    logging.info(f"训练集和验证集重叠数量: {train_val_overlap}")
    logging.info(f"训练集和测试集重叠数量: {train_test_overlap}")
    logging.info(f"验证集和测试集重叠数量: {val_test_overlap}")

    # 计算类别权重
    label_counts = pd.Series(train_labels).value_counts()
    total_samples = len(train_labels)
    class_weights = {label: total_samples / (len(label_counts) * count)
                     for label, count in label_counts.items()}

    # 转换标签为数字并应用权重
    disease_map = {
        "糖尿病": 0,
        "高血压": 1,
        "乳腺癌": 2,
        "艾滋病": 3,
        "乙肝": 4
    }

    train_labels = [disease_map[label] for label in train_labels]
    val_labels = [disease_map[label] for label in val_labels]
    test_labels = [disease_map[label] for label in test_labels]

    # 创建权重张量
    weight_tensor = torch.tensor([class_weights[label] for label in disease_map.keys()]).to(device)

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 调整 Batch Size
    train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, batch_size=32, shuffle=True)
    val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, batch_size=32, shuffle=False)
    test_dataset = MedicalDataset(test_texts, test_labels, tokenizer, batch_size=32, shuffle=False)

    # 初始化模型
    model = BertClassifier(n_classes=len(disease_map)).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)

    # 创建学习率调度器
    num_training_steps = len(train_dataset) * 10
    num_warmup_steps = num_training_steps // 10  # 增加warmup步数
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # 训练模型
    train_model(model, train_dataset, val_dataset, criterion, optimizer, scheduler,
                disease_map=disease_map, num_epochs=3, early_stopping_patience=3)

    # 评估模型
    evaluate_model(model, test_dataset, criterion, disease_map)

    # 创建输出目录
    output_dir = 'bert_medical_model'
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整的模型和tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logging.info(f"模型已保存到: {output_dir}")
    logging.info("模型训练和评估完成！")


if __name__ == "__main__":
    main()
