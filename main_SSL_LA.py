import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import pandas as pd
from data_utils_SSL import Dataset_AIGC, pad, process_Rawboost_feature
from model import Model
from tensorboardX import SummaryWriter
import random


def set_random_seed(seed, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cudnn_benchmark_toggle:
        torch.backends.cudnn.benchmark = True
    if args.cudnn_deterministic_toggle:
        torch.backends.cudnn.deterministic = True


def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y, _ in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)

    val_loss /= num_total
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()

    with open(save_path, 'w') as fh:
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                batch_out = model(batch_x)
                batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            for i in range(len(utt_id)):
                fh.write('{} {}\n'.format(utt_id[i], batch_score[i]))

    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_total = 0.0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y, _ in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    return running_loss


def split_train_val(csv_path, val_ratio=0.2):
    df = pd.read_csv(csv_path)
    # 随机打乱
    df = df.sample(frac=1).reset_index(drop=True)
    # 划分验证集
    val_size = int(len(df) * val_ratio)
    val_df = df[:val_size]
    train_df = df[val_size:]

    # 保存临时文件
    train_path = csv_path.replace('.csv', '_train_temp.csv')
    val_path = csv_path.replace('.csv', '_val_temp.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    return train_path, val_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AIGC-Speech Anti-Spoofing System')

    # 数据集参数
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Path to testing CSV file (only for evaluation)')

    # 训练参数 (严格保留原论文设置)
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE', choices=['WCE'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--freeze_features', action='store_true', default=False,
                        help='Freeze pre-trained feature extractor')

    # 模型参数
    parser.add_argument('--model_path', type=str, default=None,
                        help='Pre-trained model checkpoint path')
    parser.add_argument('--xlsr_path', type=str, default='xlsr_53_56k.pt',
                        help='Path to XLSR pre-trained model')

    # 评估参数
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='Run in evaluation mode')

    # RawBoost 数据增强参数 (保留所有原始参数)
    parser.add_argument('--algo', type=int, default=5,
                        help='RawBoost augmentation algorithm')
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    # 系统参数
    parser.add_argument('--cudnn_deterministic_toggle', action='store_false', default=True,
                        help='Disable CuDNN deterministic mode')
    parser.add_argument('--cudnn_benchmark_toggle', action='store_true', default=False,
                        help='Enable CuDNN benchmark mode')

    args = parser.parse_args()

    # 设置随机种子
    set_random_seed(args.seed, args)

    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # 创建模型
    model = Model(args, device)
    model = model.to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

    # 加载预训练模型
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Loaded pre-trained model: {args.model_path}')

    # 设置优化器 (严格保留原论文设置)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 评估模式
    if args.eval:
        if not args.test_csv:
            print("Error: Test CSV is required for evaluation")
            sys.exit(1)

        print("Running evaluation...")
        test_set = Dataset_AIGC(
            args=args,
            csv_path=args.test_csv,
            audio_dir=args.audio_dir,
            is_eval=True
        )
        if not args.eval_output:
            args.eval_output = 'eval_scores.txt'

        produce_evaluation_file(test_set, model, device, args.eval_output)
        print("Evaluation completed.")
        sys.exit(0)

    # 训练模式
    print("Preparing data...")

    # 自动划分训练集和验证集
    train_csv_temp, val_csv_temp = split_train_val(args.train_csv)

    train_set = Dataset_AIGC(
        args=args,
        csv_path=train_csv_temp,
        audio_dir=args.audio_dir,
        is_train=True,
        algo=args.algo
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=8)

    val_set = Dataset_AIGC(
        args=args,
        csv_path=val_csv_temp,
        audio_dir=args.audio_dir,
        is_train=False
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=8)

    # 训练循环
    print("Starting training...")
    model_save_path = 'models'
    os.makedirs(model_save_path, exist_ok=True)
    writer = SummaryWriter(log_dir='logs')

    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        train_loss = train_epoch(train_loader, model, args.lr, optimizer, device)
        val_loss = evaluate_accuracy(val_loader, model, device)

        print(f'Epoch {epoch + 1}/{args.num_epochs}: '
              f'Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
            print(f'Saved best model with val loss: {val_loss:.4f}')

        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(
                model_save_path, f'epoch_{epoch + 1}.pth'))

        # TensorBoard日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

    # 清理临时文件
    os.remove(train_csv_temp)
    os.remove(val_csv_temp)

    print("Training completed. Best model saved as models/best_model.pth")