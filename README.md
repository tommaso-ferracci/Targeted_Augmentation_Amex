# Targeted synthetic data generation for tabular data

Synthetic data generation has already proven successful in improving model robustness in presence of scarce or low-quality data. Based on the data valuation framework, which allows to differentiate statistically between data beneficial and detrimental to model training, we propose a novel augmentation pipeline, based on the synthetic generation of only high-value training points.

More specifically, we first demonstrate via benchmarks on real data that Shapley-based data valuation methods perform comparably with learning-based methods in hardness characterisation tasks, while offering significant theoretical advantages. Then, we show that synthetic data generators trained on only the hardest points outperform, both in quality of out-of-sample predictions and in computational efficiency, non-targeted data augmentation on a large scale credit default dataset provided by American Express and on simulated data.
