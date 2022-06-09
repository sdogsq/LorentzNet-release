# ROC Curves for Quark-gluon Tagging 

<figure>
<p align="center"><img src="./figures/QGTaggingROC_light.jpg" alt="ROC Curves for  Quark-gluon Tagging " width="40%"/></p>
<p align="center">A comparison of ROC curves between LorentzNet and other algorithms on quark-gluon tagging dataset.</p>
</figure>

## Contents
- Tagging scores given by LorentzNet and other models are saved in [scores](./scores). 

- [`ROC.py`](./ROC.py): Builds an average ROC curve from a set of trained instances of the network, complete with an error band given by $\pm \text{ std}$, i.e., the standard deviation. Adapted from [[1]](https://github.com/fizisist/LorentzGroupNetwork/tree/master/figures/ROC_curves).

## References
[1] https://github.com/fizisist/LorentzGroupNetwork/tree/master/figures/ROC_curves
