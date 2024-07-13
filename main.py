import os
import logging.config
import time
import argparse
from utils.utils import *
from utils.train_utils import *
from train import Trainer, get_dataloader_keyword

if __name__ == "__main__":
    def options():
        parser = argparse.ArgumentParser(description="Input optional guidance for training")
        parser.add_argument("--epoch", default=30, type=int, help="The number of training epoch")
        parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
        parser.add_argument("--batch", default=256, type=int, help="Training batch size")
        parser.add_argument("--step", default=20, type=int, help="Training step size")
        parser.add_argument("--gpu", default=1, type=int, help="Number of GPU device")
        parser.add_argument("--root", default="/content/KWS/dataset", type=str, help="The path of dataset")
        parser.add_argument("--dataset", default="gsc_v2", help="The name of the data set")
        parser.add_argument("--model", default="convmixer", type=str, help="models")
        parser.add_argument("--freq", default=30, type=int, help="Model saving frequency (in step)")
        parser.add_argument("--save", default="weight", type=str, help="The save name")
        parser.add_argument("--opt", default="adam", type=str, help="The optimizer")
        parser.add_argument("--sche", default="cos", type=str, help="The scheduler")
        parser.add_argument("--noise_aug", action='store_true', help="Whether to apply noise augmentation")
        parser.add_argument("--rir_aug", action='store_true', help="Whether to apply RIR augmentation")
        parser.add_argument("--musan_path", default="/content/KWS/dataset/musan", type=str, help="Path to MUSAN noise dataset")
        parser.add_argument("--rir_path", default="/content/KWS/dataset/RIRS_NOISES", type=str, help="Path to RIR dataset")
        parser.add_argument("--noise_levels", default="[0, -5, -10]", type=str, help="List of noise levels for curriculum learning")
        parser.add_argument("--patience", default=1, type=int, help="Patience for curriculum learning")
        args = parser.parse_args()
        return args

    parameters = options()

    if parameters.dataset == "gsc_v1" or parameters.dataset == "gsc_v2":
        class_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
        class_encoding = {category: index for index, category in enumerate(class_list)}

    save_path = f"{parameters.dataset}/{parameters.model}_lr{parameters.lr}_epoch{parameters.epoch}"
    logging.config.fileConfig("/content/KWS/logging.conf")
    logger = logging.getLogger()
    os.makedirs(f"logs/{parameters.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(f"[1] Select a KWS dataset ({parameters.dataset})")
    model = select_model(parameters.model, len(class_list))
    logger.info(f"[2] Select a KWS model ({parameters.model})")
    optimizer, scheduler = select_optimizer(parameters.opt, parameters.lr, model, parameters.sche)
    data_path = os.path.join(parameters.root, parameters.dataset)
    logger.info(f"[4] Load the KWS dataset from {data_path}")
    train_loader, valid_loader, test_loader = get_dataloader_keyword(
        data_path, class_list, class_encoding, parameters, noise_aug=parameters.noise_aug)
    start_time = time.time()
    trainer = Trainer(parameters, model)
    trainer.train_curriculum(optimizer=optimizer, scheduler=scheduler,
                             train_dataloader=train_loader,
                             valid_dataloader=valid_loader)
    result = trainer.model_test(test_loader)
    duration = time.time() - start_time
    logger.info(f"======== Summary =======")
    logger.info(f"{parameters.model} parameters: {parameter_number(model)}")
    logger.info(f"Total time {duration}, Avg: {duration / parameters.epoch}s")
