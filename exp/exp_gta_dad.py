import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data.data_loader_dad import (
    NASA_Anomaly,
    WADI,
    SWaT
)
from exp.exp_basic import Exp_Basic
from models.gta import GTA
from utils.metrics import optimized_metric
from utils.tools import EarlyStopping, adjust_learning_rate

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class Exp_GTA_DAD(Exp_Basic):
    def __init__(self, args):
        super(Exp_GTA_DAD, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'gta': GTA,
        }
        if self.args.model == 'gta':
            model = model_dict[self.args.model](
                self.args.num_nodes,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.num_levels,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.data,
                self.args.activation,
                self.device
            )

        return model.double()

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'SMAP': NASA_Anomaly,
            'MSL': NASA_Anomaly,
            'WADI': WADI,
            'SWaT': SWaT,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label) in enumerate(vali_loader):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)

                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = './checkpoints/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)

                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                loss = criterion(outputs, batch_y) + \
                       torch.sum(torch.abs(self.model.gt_embedding.gc_modules.logits[:, 0]))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def visualize_multi_features(self, true, pred, label, folder_path=None, name='multi_feature', num_samples=5):
        """
        多特征并行可视化（每个特征一个子图）
        """
        num_samples = min(num_samples, len(true))
        seq_len, num_features = true.shape[1], true.shape[2]

        for i in range(num_samples):
            fig, axes = plt.subplots(num_features, 1, figsize=(14, 3 * num_features), sharex=True)

            if num_features == 1:
                axes = [axes]

            for f in range(num_features):
                axes[f].plot(true[i, :, f], label='True')
                axes[f].plot(pred[i, :, f], label='Pred')

                # 标记异常点
                anomalies = np.where(label[i] == 1)[0]
                for a in anomalies:
                    axes[f].axvline(x=a, color='r', linestyle='--', alpha=0.6)

                axes[f].set_title(f"Feature {f + 1}")
                axes[f].legend()

            plt.tight_layout()
            if folder_path is not None:
                save_path = os.path.join(folder_path, f'{name}_sample_{i}.pdf')
                plt.savefig(save_path, bbox_inches='tight')
            else:
                plt.show()
            plt.close()

    def plot_threshold_analysis(self, errors, labels, folder_path=None, name='threshold_analysis'):
        """
        绘制误差分布 + 阈值性能曲线
        """
        total_errors = errors.flatten()
        total_labels = labels.flatten()

        normal_errors = total_errors[total_labels == 0]
        anomaly_errors = total_errors[total_labels == 1]

        thresholds = np.linspace(np.min(total_errors), np.max(total_errors), 100)
        precisions, recalls, f1_scores = [], [], []

        from sklearn.metrics import classification_report

        for t in thresholds:
            binary_preds = (total_errors > t).astype(int)
            report = classification_report(total_labels, binary_preds, output_dict=True)
            precisions.append(report.get('Anomaly', {}).get('precision', 0))
            recalls.append(report.get('Anomaly', {}).get('recall', 0))
            f1_scores.append(report.get('Anomaly', {}).get('f1-score', 0))

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(normal_errors, bins=50, kde=True, label='Normal', color='g', alpha=0.5)
        sns.histplot(anomaly_errors, bins=50, kde=True, label='Anomaly', color='r', alpha=0.5)
        plt.axvline(best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
        plt.title("Error Distribution (Normal vs Anomaly)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1 Score')
        plt.axvline(best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
        plt.title("Threshold vs Performance")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.legend()

        plt.tight_layout()
        if folder_path is not None:
            save_path = os.path.join(folder_path, f'{name}.pdf')
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []
        labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label) in enumerate(test_loader):
                batch_x = batch_x.double().to(self.device)
                batch_y = batch_y.double().to(self.device)
                batch_x_mark = batch_x_mark.double().to(self.device)
                batch_y_mark = batch_y_mark.double().to(self.device)

                if i % 100 == 0:
                    print(f'i:{i},batch_y shape:{batch_y.shape}, batch_label shape:{batch_label.shape}')

                outputs = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                batch_label = batch_label.long().detach().numpy()

                preds.append(pred)
                trues.append(true)
                labels.append(batch_label)

        preds = np.array(preds)
        trues = np.array(trues)
        labels = np.array(labels)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        labels = labels.reshape(-1, labels.shape[-1])

        print('test shape:', preds.shape, trues.shape, labels.shape)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = optimized_metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        from sklearn.metrics import classification_report

        assert preds.shape == trues.shape, "preds and trues shape mismatch"
        diff_tensor = preds - trues
        abs_diff_tensor = np.abs(diff_tensor)
        mean_abs_diff_tensor = np.mean(abs_diff_tensor, axis=-1)

        assert mean_abs_diff_tensor.shape == labels.shape, "标签和预测误差的时间步不一致"

        all_mean_abs_errors = np.mean(np.abs(preds - trues), axis=(0, 2))
        threshold = max(np.percentile(all_mean_abs_errors, 95), 1e-4)

        binary_preds = (mean_abs_diff_tensor > threshold).astype(int).flatten()
        binary_labels = labels.flatten()

        class_report = classification_report(binary_labels, binary_preds,
                                             target_names=['Normal', 'Anomaly'],
                                             output_dict=True)
        accuracy = class_report['accuracy']
        precision = class_report.get('Anomaly', {}).get('precision', 0.0)
        recall = class_report.get('Anomaly', {}).get('recall', 0.0)
        f1_score = class_report.get('Anomaly', {}).get('f1-score', 0.0)

        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

        metrics = np.array([mae, mse, rmse, mape, mspe, accuracy, precision, recall, f1_score])
        np.save(os.path.join(folder_path, 'metrics.npy'), metrics)
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        np.save(os.path.join(folder_path, 'label.npy'), labels)

        # 新增可视化功能
        self.visualize_multi_features(trues, preds, labels, folder_path=folder_path, name='multi_feature', num_samples=5)
        self.plot_threshold_analysis(mean_abs_diff_tensor, labels, folder_path=folder_path, name='threshold_analysis')

        return