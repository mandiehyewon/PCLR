import numpy as np

def do_bootstrap(pred_vals, trues, threshold=0.5, n=1000):
    auc_list = []
    apr_list = []
    acc_list = []
    f1_list = []
    
    preds = np.array(pred_vals > threshold).astype('float32')
    
    rng = np.random.RandomState(seed=1)
    for _ in range(n):
        idxs = rng.choice(len(trues), size=len(trues), replace=True)
        pred_arr= preds[idxs]
        true_arr = trues[idxs]
        pred_val_arr = pred_vals[idxs]

        auc = roc_auc_score(true_arr, pred_arr)
        apr = average_precision_score(true_arr, pred_arr)
        acc = accuracy_score(true_arr, np.concatenate(pred_arr))
        f1 = f1_score(true_arr, pred_arr)

        auc_list.append(auc)
        apr_list.append(apr)
        acc_list.append(acc)
        f1_list.append(f1)

    return np.array(auc_list), np.array(apr_list), np.array(acc_list), np.array(f1_list)

def confidence_interval(values, alpha=0.95):
    lower = np.percentile(values, (1-alpha)/2 * 100)
    upper = np.percentile(values, (alpha + (1-alpha)/2) * 100)
    return lower, upper