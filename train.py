import os
import logging
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import *
from utils.utils import *
import torchaudio

logger = logging.getLogger()

def get_dataloader_keyword(data_path, class_list, class_encoding, parameters, noise_aug=False):
    if len(class_list) == 0:
        raise ValueError("The class list is empty!")
    batch_size = parameters.batch
    train_json = f"{data_path}/train_manifest.json"
    valid_json = f"{data_path}/validation_manifest.json"
    test_json = f"{data_path}/test_manifest.json"
    
    train_dataset = SpeechCommandDataset(data_path, train_json, True, class_list, class_encoding, noise_aug=noise_aug, musan_path=parameters.musan_path, rir_path=parameters.rir_path)
    valid_dataset = SpeechCommandDataset(data_path, valid_json, False, class_list, class_encoding, musan_path=parameters.musan_path, rir_path=parameters.rir_path)
    test_dataset = SpeechCommandDataset(data_path, test_json, False, class_list, class_encoding, musan_path=parameters.musan_path, rir_path=parameters.rir_path)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_dataloader, valid_dataloader, test_dataloader

class Trainer:
    def __init__(self, opt, model):
        self.opt = opt
        self.lr = opt.lr
        self.step = opt.step
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.model = model
        self.device, self.device_list = prepare_device(opt.gpu)
        self.model.to(self.device)
        if len(self.device_list) > 1:
            print(f">>>   Available GPU device: {self.device_list}")
            self.model = nn.DataParallel(self.model)
        self.best_acc = 0.0
        self.best_model = model
        self.criterion = nn.CrossEntropyLoss()
        self.loss_name = {
            "train_loss": 0.0, "train_accuracy": 0.0, "train_total": 0, "train_correct": 0,
            "valid_loss": 0.0, "valid_accuracy": 0.0, "valid_total": 0, "valid_correct": 0}
        
        # 初始化 musan_noise_dataset 和 rir_dataset
        self.musan_noise_dataset = self.load_noise_dataset(opt.musan_path)
        self.rir_dataset = self.load_rir_dataset(opt.rir_path)
        
    def load_noise_dataset(self, musan_path):
        noise_dataset = []
        for root, _, filenames in os.walk(musan_path):
            for fn in filenames:
                if fn.endswith('.wav'):
                    noise_dataset.append(os.path.join(root, fn))
        logger.info(f"Loaded {len(noise_dataset)} noise files from {musan_path}")
        return noise_dataset

    def load_rir_dataset(self, rir_path):
        rir_dataset = []
        for root, _, filenames in os.walk(rir_path):
            for fn in filenames:
                if fn.endswith('.wav'):
                    rir_dataset.append(os.path.join(root, fn))
        logger.info(f"Loaded {len(rir_dataset)} RIR files from {rir_path}")
        return rir_dataset

    def model_save(self):
        save_directory = os.path.join("./model_save", self.opt.save)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        if self.loss_name["valid_accuracy"] > self.best_acc:
            self.best_acc = self.loss_name["valid_accuracy"]
            self.best_model = self.model
            logger.info(f"Saving the best model with accuracy {self.best_acc:.4f}")
            torch.save(self.model.state_dict(), os.path.join(save_directory, f"best.pt"))
        if (self.epo + 1) == self.epoch:
            torch.save(self.model.state_dict(), os.path.join(save_directory, "last.pt"))

    def add_noise(self, data, noise_level):
        if noise_level == 'clean':
            return data
        noise_index = torch.randint(0, len(self.musan_noise_dataset), size=(1,)).item()
        noise, _ = torchaudio.load(self.musan_noise_dataset[noise_index])
        noise = noise.to(data.device)  # 将噪声移动到与数据相同的设备
        if noise.shape[1] < data.shape[1]:
            noise = F.pad(noise, [0, data.shape[1] - noise.shape[1]])
        else:
            offset = torch.randint(0, noise.shape[1] - data.shape[1] + 1, size=(1,)).item()
            noise = noise.narrow(1, offset, data.shape[1])
        noise = noise * 10 ** (noise_level / 20.0)
        return data + noise

    def augment_with_rir(self, data):
        rir_index = torch.randint(0, len(self.rir_dataset), size=(1,)).item()
        rir, _ = torchaudio.load(self.rir_dataset[rir_index])
        rir = rir.to(data.device)  # 将RIR移动到与数据相同的设备
        rir = rir / torch.norm(rir, p=2)
        if data.dim() == 2:  # 如果数据是2D的，则添加一个批次维度
            data = data.unsqueeze(0)
        data = torch.nn.functional.conv1d(data, rir.unsqueeze(1), padding=rir.shape[1] // 2)
        return data.squeeze(0)  # 移除批次维度

    def normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            return [0.0 for _ in values]
        return [(val - min_val) / (max_val - min_val) for val in values]

    def train_curriculum(self, optimizer, scheduler, train_dataloader, valid_dataloader):
        noise_levels = [
            ['clean'],          # 干净样本
            ['clean', 0],         # 噪声级别 0 和 干净样本
            ['clean', 0, -5],     # 噪声级别 0 和 -5 和 干净样本    
            ['clean', 0, -5, -10] # 噪声级别 0, -5, 和 -10 和 干净样本
        ]
        best_criterion = -float('inf')
        patience = self.opt.patience
        stages = len(noise_levels) # 阶段数
        train_length, valid_length = len(train_dataloader), len(valid_dataloader)

        for stage in range(stages):
            logger.info(f"Starting stage {stage+1}/{stages}")  # 记录当前阶段的日志
            epoch_accuracies = []
            epoch_losses = []
            patience_counter = 0  # 每个阶段重置耐心计数器

            for self.epo in range(self.epoch): 
                logger.info(f"Starting epoch {self.epo+1}/{self.epoch} in stage {stage+1}")  # 记录当前 epoch 的日志
                self.loss_name.update({key: 0 for key in self.loss_name})
                self.model.cuda(self.device)
                self.model.train()
                for batch_idx, (waveform, labels) in tqdm(enumerate(train_dataloader), position=0, total=len(train_dataloader)):
                    waveform, labels = waveform.to(self.device), labels.to(self.device)
                    logger.info(f"Applying noise level(s) {noise_levels[stage]} at stage {stage}, epoch {self.epo+1}")  # 记录噪声增强信息
                    # for noise_level in noise_levels[stage]:
                    #     waveform = self.add_noise(waveform, noise_level)
                    # Randomly select a noise level from the current stage's noise levels 随机选择当前阶段的噪声级别，而不是循环，保证均匀分布
                    selected_noise_level = random.choice(noise_levels[stage])
                    waveform = self.add_noise(waveform, selected_noise_level)

                    optimizer.zero_grad()
                    logits,labels = self.model(waveform, labels)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    self.loss_name["train_loss"] += loss.item() / train_length
                    _, predict = torch.max(logits.data, 1)
                    self.loss_name["train_total"] += labels.size(0)
                    self.loss_name["train_correct"] += (predict == labels).sum().item()
                    self.loss_name["train_accuracy"] = self.loss_name["train_correct"] / self.loss_name["train_total"]

                logger.info(f"Epoch {self.epo+1}/{self.epoch} in stage {stage+1} completed. Training accuracy: {self.loss_name['train_accuracy']:.4f}")

                self.model.eval()
                for batch_idx, (waveform, labels) in enumerate(valid_dataloader):
                    with torch.no_grad():
                        waveform, labels = waveform.to(self.device), labels.to(self.device)
                        logits = self.model(waveform)
                        loss = self.criterion(logits, labels)
                        self.loss_name["valid_loss"] += loss.item() / valid_length
                        _, predict = torch.max(logits.data, 1)
                        self.loss_name["valid_total"] += labels.size(0)
                        self.loss_name["valid_correct"] += (predict == labels).sum().item()
                        self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]
                
                logger.info(f"Epoch {self.epo+1}/{self.epoch} in stage {stage+1} completed. Validation accuracy: {self.loss_name['valid_accuracy']:.4f}")
                
                scheduler.step()
                self.model_save()

                epoch_accuracy = 100 * self.loss_name['train_accuracy']
                epoch_loss = self.loss_name['train_loss']
                epoch_accuracies.append(epoch_accuracy)
                epoch_losses.append(epoch_loss)

                norm_accuracy = self.normalize(epoch_accuracies)
                norm_loss = self.normalize(epoch_losses)
                c = norm_accuracy[-1] - norm_loss[-1]

                if c > best_criterion:
                    best_criterion = c
                    patience_counter = 0  # 重置耐心计数器
                else:
                    patience_counter += 1

                logger.info(f"Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info(f"Patience counter reached in stage {stage+1}, moving to next stage")  # 记录耐心计数器日志
                    break

        logger.info("Completed all stages of curriculum learning")

    def model_test(self, test_dataloader):
        self.best_model.eval()
        test_length = len(test_dataloader)
        self.loss_name.update({key: 0 for key in self.loss_name})
        for batch_idx, (waveform, labels) in enumerate(test_dataloader):
            with torch.no_grad():
                waveform, labels = waveform.to(self.device), labels.to(self.device)
                logits = self.best_model(waveform)
                loss = self.criterion(logits, labels)
                self.loss_name["valid_loss"] += loss.item() / test_length
                _, predict = torch.max(logits.data, 1)
                self.loss_name["valid_total"] += labels.size(0)
                self.loss_name["valid_correct"] += (predict == labels).sum().item()
                self.loss_name["valid_accuracy"] = self.loss_name["valid_correct"] / self.loss_name["valid_total"]
                self.loss_name["f1_score"] = f1_score(labels.cpu().numpy(), predict.cpu().numpy(), average='macro')
        logger.info(
            f"test_loss {self.loss_name['valid_loss']:.4f} "
            f"| test_acc {self.loss_name['valid_accuracy']:.4f}"
            f"| f1_score {self.loss_name['f1_score']:.4f}"
        )
        return self.loss_name

