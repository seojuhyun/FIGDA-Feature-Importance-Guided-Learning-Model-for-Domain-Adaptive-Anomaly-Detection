#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIGDA Main Execution Script

Author: Juhyun Seo (2026)
PAKDD 2026

This script runs the FIGDA model using the implementation
provided in run_figda.py.
"""

import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from run_figda import FIGDAClassifier

# ------------------------------------------------------------
# Reproducibility settings
# ------------------------------------------------------------

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

data_path = "./dataset/celeba_baldvsnonbald_normalised.csv"
data = pd.read_csv(data_path)

# ------------------------------------------------------------
# Run FIGDA
# ------------------------------------------------------------

model = FIGDAClassifier(
    data=data,
    dataset_name="CelebA",
    output_path="./results"
)

model.forward(train=True)
