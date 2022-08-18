from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

pred_Y = predictions.cpu().numpy()
test_Y = targets.cpu().numpy()

all_labels = [i for i in range(14)]

fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:, idx], pred_Y[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()
fig.savefig('auc_roc_own_split.png')

print('ROC auc score: {:.3f}'.format(roc_auc_score(test_Y, pred_Y)))