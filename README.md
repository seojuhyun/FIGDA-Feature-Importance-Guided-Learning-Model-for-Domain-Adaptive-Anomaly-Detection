# FIGDA: Feature Importance Guided Learning Model for Domain-Adaptive Anomaly Detection

Official implementation of:

**FIGDA: Feature Importance Guided Learning Model for Domain-Adaptive Anomaly Detection**  
Juhyun Seo  
PAKDD 2026 (Main Track, Oral Presentation)

---

## 📌 Overview

FIGDA integrates feature importance (FI) into deviation-based anomaly detection by:

- FI-guided weight initialization  
- FI-regularized loss  
- Domain-adaptive training strategy  

The framework enhances representation-level separability while maintaining stable performance across heterogeneous datasets.

---

## 🧠 Framework Summary

FIGDA extends DevNet (Pang et al., KDD 2019) by incorporating feature importance extracted from XGBoost into:

1. First-layer weight initialization  
2. A feature-importance regularization term during training  

This structured guidance improves anomaly ranking consistency under weak supervision.

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- PyTorch
- NumPy
- pandas
- scikit-learn
- xgboost
- matplotlib

---

## 🔧 Hyperparameter Settings

- `alpha_first_layer` : FI-based initialization strength  
- `lambda_fi` : FI regularization coefficient  

⚠️ These hyperparameters are dataset-dependent.  
Users are encouraged to tune them according to dataset characteristics.

---

## 📝 License

This implementation is provided for research and academic purposes.

Portions of the code are derived from:

Pang, G., Shen, C., & van den Hengel, A. (2019).  
*Deep Anomaly Detection with Deviation Networks.*  
Proceedings of KDD 2019.

The original DevNet license terms apply to inherited components.

---

## 👤 Author

Juhyun Seo  
PAKDD 2026
