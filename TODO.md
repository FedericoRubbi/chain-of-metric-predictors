## Testing
### What architectures to test?
- baseline MLP trained with BP (MLP)
- baseline MLP trained with forward-forward algorithm (FF)
- baseline MLP trained with collaborative forward-forward (CFF)
- MLP trained with greedy BP (GMLP)


### What to test?
metrics provide insight for the following questions:
- are later layers getting more predictive?
- are layers compressing?
- are layers genuinely different from previous layers?

### What data to collect for each epoch?

MLP:
- cosine-to-label anchors (per layer) for alignment: \;\overline{\cos(z_\ell, a_{y})} and margin: \;\overline{\cos(z_\ell, a_{y}) - \max_{k\neq y}\cos(z_\ell, a_{k})}
- layerwise accuracy on test set
- layerwise cross-entropy with GT
- layerwise ACE regularizer, namely cross-entropy between consecutive softmax heads (averaged over a batch)
- functional entropy of each layer
- MI between input and each layer's output
- gaussian-entropy proxy of EMA of activations as \widehat{H}(Z_\ell)\propto \sum_i \tfrac12 \log \hat\sigma_i^2
- F1 score, accuracy One-shot linear probe (ridge, closed form), fit W = (X^\top X + \lambda I)^{-1}X^\top Y on one held-out minibatch
- participance ration from the activation covariance C on a minibatch (or diagonal approx): \mathrm{PR}=\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
- linear CKA (minibatch): \text{CKA}(Z_\ell,Z_{\ell+1}) = \frac{\|Z_\ell^\top Z_{\ell+1}\|F^2}{\|Z\ell^\top Z_\ell\|F \cdot \|Z{\ell+1}^\top Z_{\ell+1}\|_F}
- R^2 of ridge regression Z_{\ell+1}\approx Z_\ell W on a minibatch

GMLP:
- cosine-to-label anchors (per layer) for alignment: \;\overline{\cos(z_\ell, a_{y})} and margin: \;\overline{\cos(z_\ell, a_{y}) - \max_{k\neq y}\cos(z_\ell, a_{k})}
- layerwise accuracy on test set
- layerwise cross-entropy with GT
- layerwise ACE regularizer, namely cross-entropy between consecutive softmax heads (averaged over a batch)
- functional entropy of each layer
- MI between input and each layer's output
- gaussian-entropy proxy of EMA of activations as \widehat{H}(Z_\ell)\propto \sum_i \tfrac12 \log \hat\sigma_i^2
- F1 score, accuracy One-shot linear probe (ridge, closed form), fit W = (X^\top X + \lambda I)^{-1}X^\top Y on one held-out minibatch
- participance ration from the activation covariance C on a minibatch (or diagonal approx): \mathrm{PR}=\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
- linear CKA (minibatch): \text{CKA}(Z_\ell,Z_{\ell+1}) = \frac{\|Z_\ell^\top Z_{\ell+1}\|F^2}{\|Z\ell^\top Z_\ell\|F \cdot \|Z{\ell+1}^\top Z_{\ell+1}\|_F}
- R^2 of ridge regression Z_{\ell+1}\approx Z_\ell W on a minibatch