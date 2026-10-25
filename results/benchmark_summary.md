# FedMol — benchmark summary

## classification

| regime | endpoint | metric | value [95% CI] |
|---|---|---|---|
| isolated | ames | pr_auc | 0.864 [0.841–0.890] |
| federated | ames | pr_auc | 0.900 [0.881–0.922] |
| centralized | ames | pr_auc | 0.901 [0.883–0.923] |
| isolated | herg | pr_auc | 0.797 [0.780–0.814] |
| federated | herg | pr_auc | 0.840 [0.825–0.854] |
| centralized | herg | pr_auc | 0.840 [0.826–0.855] |

## regression

| regime | endpoint | metric | value [95% CI] |
|---|---|---|---|
| isolated | solubility | rmse | 1.630 [1.541–1.720] |
| federated | solubility | rmse | 1.580 [1.491–1.676] |
| centralized | solubility | rmse | 1.498 [1.415–1.589] |
| isolated | logp | rmse | 0.965 [0.912–1.017] |
| federated | logp | rmse | 0.892 [0.836–0.950] |
| centralized | logp | rmse | 0.871 [0.817–0.926] |

## Honest readings

### Coverage (applicability domain)

On molecules 0.30–0.50 Tanimoto-far from a partner's own chemistry, isolated partners average nan accuracy vs federated nan — no clear FL edge here.

### Expectation check

Federated should land between isolated and centralized-oracle, with the largest gain on the sparsest endpoint (LogP, ~4.2k rows). If isolated matches or beats FL everywhere, reframe: FL matched it while no partner ever saw the others' chemistry.
