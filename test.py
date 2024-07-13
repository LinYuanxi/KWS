import os
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from utils.utils import prepare_device
from utils.data_loader import SpeechCommandDataset
from utils.train_utils import select_model
from sklearn.metrics import f1_score

def setup_logger():
    logging.basicConfig(
        format="[%(levelname)s] %(filename)s:%(lineno)d > %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()
    return logger

def load_best_model(opt, model):
    model_path = os.path.join("./model_save", opt.save, "best.pt")
    model.load_state_dict(torch.load(model_path, map_location=opt.device))
    model.to(opt.device)
    model.eval()
    logger.info(f"Loaded best model from {model_path}")
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (waveform, labels) in enumerate(dataloader):
            waveform, labels = waveform.to(device), labels.to(device)
            logits = model(waveform)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    logger.info(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")
    return avg_loss, accuracy, f1

def options():
    parser = argparse.ArgumentParser(description="Test the trained model on the test dataset")
    parser.add_argument("--batch", default=256, type=int, help="Batch size for testing")
    parser.add_argument("--gpu", default=1, type=int, help="Number of GPU device")
    parser.add_argument("--root", default="/content/KWS/dataset", type=str, help="The path of dataset")
    parser.add_argument("--dataset", default="/content/dataset/20dB/augmented_audio_dataset_20dB/gsc_v2", help="The name of the dataset")
    parser.add_argument("--model", default="convmixer", type=str, help="Model type")
    parser.add_argument("--save", default="weight", type=str, help="The save name")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logger = setup_logger()
    parameters = options()
    device, device_list = prepare_device(parameters.gpu)
    parameters.device = device

    logger.info(f"Selected device: {device}")

    # 检查文件路径是否正确
    test_manifest_path = os.path.join(parameters.root, parameters.dataset, "test_manifest.json")
    if not os.path.exists(test_manifest_path):
        logger.error(f"Test manifest file not found at {test_manifest_path}")
        raise FileNotFoundError(f"Test manifest file not found at {test_manifest_path}")

    # 加载模型
    model = select_model(parameters.model, 12)  # 假设 GSC 数据集有 12 个类别
    model = load_best_model(parameters, model)

    # 创建数据集和数据加载器
    test_dataset = SpeechCommandDataset(
        parameters.root,
        test_manifest_path,
        False,
        class_list=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"],
        class_encoding={category: index for index, category in enumerate(["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"])}
    )

    test_loader = DataLoader(test_dataset, batch_size=parameters.batch, shuffle=False, drop_last=True)

    # 添加更多日志信息，确保评估过程执行
    logger.info("Starting evaluation of the model")
    avg_loss, accuracy, f1 = evaluate_model(model, test_loader, device)
    logger.info(f"Final Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
