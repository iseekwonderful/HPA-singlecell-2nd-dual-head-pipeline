import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from path import Path
import os
from sklearn.metrics import accuracy_score, roc_auc_score


def macro_multilabel_auc(label, pred, gpu=-1):
    aucs = []
    for i in range(11):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    if gpu == 0:
        print(np.round(aucs, 4))
    return np.mean(aucs)


METRIC_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

CFG = {
    'image_target_cols': [
        'pe_present_on_image',  # only image level
    ],

    'exam_target_cols': [
        'negative_exam_for_pe',  # exam level
        # 'qa_motion',
        # 'qa_contrast',
        # 'flow_artifact',
        'rv_lv_ratio_gte_1',  # exam level
        'rv_lv_ratio_lt_1',  # exam level
        'leftsided_pe',  # exam level
        'chronic_pe',  # exam level
        # 'true_filling_defect_not_pe',
        'rightsided_pe',  # exam level
        'acute_and_chronic_pe',  # exam level
        'central_pe',  # exam level
        'indeterminate'  # exam level
    ],

    'image_weight': 0.07361963,
    'exam_weights': [0.0736196319, 0.2346625767, 0.0782208589, 0.06257668712, 0.1042944785, 0.06257668712, 0.1042944785,
                     0.1877300613, 0.09202453988],
}


def rsna_torch_wloss(CFG, y_true_img, y_true_exam, y_pred_img, y_pred_exam, chunk_sizes):
    # transform into torch tensors
    y_true_img, y_true_exam, y_pred_img, y_pred_exam = torch.tensor(y_true_img, dtype=torch.float32), torch.tensor(
        y_true_exam, dtype=torch.float32), torch.tensor(y_pred_img, dtype=torch.float32), torch.tensor(y_pred_exam,
                                                                                                       dtype=torch.float32)

    # split into chunks (each chunks is for a single exam)
    y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks = torch.split(y_true_img, chunk_sizes,
                                                                                               dim=0), torch.split(
        y_true_exam, chunk_sizes, dim=0), torch.split(y_pred_img, chunk_sizes, dim=0), torch.split(y_pred_exam,
                                                                                                   chunk_sizes, dim=0)

    label_w = torch.tensor(CFG['exam_weights']).view(1, -1)
    img_w = CFG['image_weight']
    bce_func = torch.nn.BCELoss(reduction='none')

    total_loss = torch.tensor(0, dtype=torch.float32)
    total_weights = torch.tensor(0, dtype=torch.float32)

    for i, (y_true_img_, y_true_exam_, y_pred_img_, y_pred_exam_) in enumerate(
            zip(y_true_img_chunks, y_true_exam_chunks, y_pred_img_chunks, y_pred_exam_chunks)):
        exam_loss = bce_func(y_pred_exam_[0, :], y_true_exam_[0, :])
        exam_loss = torch.sum(exam_loss * label_w, 1)[
            0]  # Kaggle uses a binary log loss equation for each label and then takes the mean of the log loss over all labels.

        image_loss = bce_func(y_pred_img_, y_true_img_)
        img_num = chunk_sizes[i]
        qi = torch.sum(y_true_img_) / img_num
        image_loss = torch.sum(img_w * qi * image_loss)

        total_loss += exam_loss + image_loss
        total_weights += label_w.sum() + img_w * qi * img_num
        # print(exam_loss, image_loss, img_num);assert False

    final_loss = total_loss / total_weights
    return final_loss


def rsna_weight_loss_image_only(df, truth):
    '''
    Fill the prediction of experiment level with mean value
    For stage1 model only!

    '''
    path = METRIC_FILE_PATH / '..' / 'dataloaders/split/naive.full.stratified.5.fold.csv.zip'
    train = pd.read_csv(path)
    exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
    df[CFG['exam_target_cols']] = exam_label_mean.values
    with torch.no_grad():
        loss = rsna_torch_wloss(CFG, truth[CFG['image_target_cols']].values, truth[CFG['exam_target_cols']].values,
                                df[CFG['image_target_cols']].values, df[CFG['exam_target_cols']].values,
                                list(truth.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))

        return loss.item()


def rsna_weight_loss_full(df, truth):
    with torch.no_grad():
        loss = rsna_torch_wloss(CFG, truth[CFG['image_target_cols']].values, truth[CFG['exam_target_cols']].values,
                                df[CFG['image_target_cols']].values, df[CFG['exam_target_cols']].values,
                                list(truth.groupby('StudyInstanceUID', sort=False)['SOPInstanceUID'].count()))

        return loss.item()


if __name__ == '__main__':
    split_df = pd.read_csv(METRIC_FILE_PATH / '..' / 'dataloaders/split/naive.full.stratified.5.fold.csv.zip')
    print(split_df.shape)
    train = split_df[split_df.fold != 0]
    valid = split_df[split_df.fold == 0]
    predicted = valid.copy()
    predicted['pe_present_on_image'] = train['pe_present_on_image'].mean()
    print(predicted.head())
    print(rsna_weight_loss_image_only(predicted, valid))
    exam_label_mean = train[CFG['exam_target_cols']].mean(axis=0)
    predicted[CFG['exam_target_cols']] = exam_label_mean.values
    print(rsna_weight_loss_full(predicted, valid))
