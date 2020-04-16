import numpy as np
from scipy import stats


def kl(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    new_q = q[q != 0]
    new_p = p[q != 0]
    return np.sum(np.where(new_p != 0, new_p * np.log(new_p / new_q), 0))


def quant(pred_dist, action_set):
    """
    :param num_classes: int
    :param pred_dist: ndarray [num, num_classes] for predicted categories
    :return: I-score, inter-entropy, class-aware inter-entropy, intra-entropy
    """
    # get the histogram of gt and pred_hist
    overall_dist = np.mean(pred_dist, axis=0)

    # get the predicted_class
    predicted_class = np.argmax(pred_dist, axis=1)

    klds = []
    Intra_Es = []
    class_Intra_Es = {}
    for idx in range(len(pred_dist)):
        intra_E = stats.entropy(pred_dist[idx])
        klds.append(kl(pred_dist[idx], overall_dist))
        Intra_Es.append(intra_E)
        # get action_class
        action_class = action_set[predicted_class[idx] % len(action_set)]
        class_Intra_E = class_Intra_Es.get(action_class, [])
        class_Intra_E.append(intra_E)
        class_Intra_Es[action_class] = class_Intra_E

    I_score = np.exp(np.mean(klds))
    Intra_E = np.mean(Intra_Es)
    for k, v in class_Intra_Es.items():
        class_Intra_Es[k] = float(np.mean(v))
    Inter_E = stats.entropy(overall_dist)
    return float(I_score), float(Intra_E), Inter_E, class_Intra_Es
