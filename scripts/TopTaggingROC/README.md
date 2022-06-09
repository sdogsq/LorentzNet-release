# ROC Curves for Top Tagging 

<figure>
<p align="center"><img src="./figures/TopTaggingROC_light.jpg" alt="ROC Curves for  Top Tagging " width="40%"/></p>
<p align = "center">A comparison of ROC curves between LorentzNet and other algorithms on top tagging dataset.</p>
</figure>

## Contents
- Tagging scores given by LorentzNet and other models are saved in [scores](./scores). Scores for other top-tagging methods are taken from [[2]](https://arxiv.org/abs/1902.09914).

- [`ROC.py`](./ROC.py): Builds an average ROC curve from a set of trained instances of the network, complete with an error band given by $\pm \text{ std}$, i.e., the standard deviation. Adapted from [[1]](https://github.com/fizisist/LorentzGroupNetwork/tree/master/figures/ROC_curves).

## References
[1] https://github.com/fizisist/LorentzGroupNetwork/tree/master/figures/ROC_curves

[2] Kasieczka, Gregor, et al. "The Machine Learning landscape of top taggers." SciPost Physics 7.1 (2019): 014.
