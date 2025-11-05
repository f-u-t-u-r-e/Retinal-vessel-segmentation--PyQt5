import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc
from numpy import mean

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def binary_confusion_matrix(prediction, groundtruth):
    """
    计算混淆矩阵
    
    Returns:
        TN: 真负例
        FP: 假正例
        FN: 假负例
        TP: 真正例
    """
    TP = np.float64(np.sum((prediction == 1) & (groundtruth == 1)))
    FP = np.float64(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float64(np.sum((prediction == 0) & (groundtruth == 1)))
    TN = np.float64(np.sum((prediction == 0) & (groundtruth == 0)))
    
    return TN, FP, FN, TP


def evaluate(pred_dir, label_dir, manual_type='1st', img_size=(512, 512), save_roc=True):
    """
    评估模型性能
    
    Args:
        pred_dir: 预测结果目录
        label_dir: 标签目录
        manual_type: 标签类型 ('1st' 或 '2nd')
        img_size: 图像尺寸
        save_roc: 是否保存ROC曲线
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    file_list = os.listdir(pred_dir)
    
    metrics = {
        'precisions': [],
        'recalls': [],
        'accuracys': [],
        'f1_scores': [],
        'f2_scores': [],
        'dices': [],
        'iou1s': [],
        'iou0s': [],
        'mean_ious': []
    }
    
    fprs = np.array([0, 0, 0]).astype(np.float64)
    tprs = np.array([0, 0, 0]).astype(np.float64)
    roc_aucs = np.array([0]).astype(np.float64)
    
    for f in file_list:
        pred = np.array(Image.open(os.path.join(pred_dir, f)).resize(
            img_size, Image.Resampling.BILINEAR)).flatten()
        
        f_label = f[:-9] + f'_manual{manual_type[-1]}.gif'
        label = np.array(Image.open(os.path.join(label_dir, f_label)).resize(
            img_size, Image.Resampling.BILINEAR)).flatten()
        label[label > 0] = 1
        
        TN, FP, FN, TP = binary_confusion_matrix(pred, label)
        
        # 计算各项指标
        precision = float(TN) / (float(TN + FP) + 1e-6)
        recall = float(TP) / (float(TP + FN) + 1e-6)
        accuracy = float(TP + TN) / (float(TP + FP + FN + TN) + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        f2_score = 2 * float(TP) / (float(2 * TP + FP + FN) + 1e-6)
        dice = 2 * float(TP) / (float(FP + 2 * TP + FN) + 1e-6)
        iou1 = float(TP) / (float(FP + TP + FN) + 1e-6)
        iou0 = float(TN) / (float(FP + TN + FN) + 1e-6)
        mean_iou = (iou1 + iou0) / 2
        
        metrics['precisions'].append(precision)
        metrics['recalls'].append(recall)
        metrics['accuracys'].append(accuracy)
        metrics['f1_scores'].append(f1_score)
        metrics['f2_scores'].append(f2_score)
        metrics['dices'].append(dice)
        metrics['iou1s'].append(iou1)
        metrics['iou0s'].append(iou0)
        metrics['mean_ious'].append(mean_iou)
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
        fprs += fpr
        tprs += tpr
        roc_aucs += auc(fpr, tpr)
    
    # 绘制ROC曲线
    if save_roc:
        fprs = fprs / len(file_list)
        tprs = tprs / len(file_list)
        roc_aucs = roc_aucs / len(file_list)
        
        plt.figure()
        plt.plot(fprs, tprs, 'k--', label=f'ROC (area = {roc_aucs[0]:.4f})', lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        os.makedirs('./roc', exist_ok=True)
        plt.savefig(f'./roc/{manual_type}_roc.png')
        plt.close()
    
    # 打印结果
    print(f"\n========== {manual_type.upper()} Manual 评估结果 ==========")
    print(f'精准率 (Sp):        {mean(metrics["precisions"]):.4f}')
    print(f'敏感度 (Se):        {mean(metrics["recalls"]):.4f}')
    print(f'准确率 (Acc):       {mean(metrics["accuracys"]):.4f}')
    print(f'F1 Score:          {mean(metrics["f1_scores"]):.4f}')
    print(f'F2 Score:          {mean(metrics["f2_scores"]):.4f}')
    print(f'Dice系数:          {mean(metrics["dices"]):.4f}')
    print(f'IoU (前景):        {mean(metrics["iou1s"]):.4f}')
    print(f'IoU (背景):        {mean(metrics["iou0s"]):.4f}')
    print(f'平均IoU:           {mean(metrics["mean_ious"]):.4f}')
    print("=" * 50)
    
    return {k: mean(v) for k, v in metrics.items()}


if __name__ == "__main__":
    # 评估第一组标注
    eval_1st = evaluate("./data/test/outputs", "./data/test/1st_manual/", manual_type='1st')
    
    # 评估第二组标注
    eval_2nd = evaluate("./data/test/outputs", "./data/test/2nd_manual/", manual_type='2nd')
