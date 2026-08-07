# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_nmx_chip_train_cfg import (
    va_nmx_chip_episode109_overfit_cfg,
    va_nmx_chip_train_cfg,
    va_nmx_fold_towel_train_mot_fastwam_cfg,
    va_nmx_chip_train_fastwam_cfg,
    va_nmx_chip_train_mot_cfg,
    va_nmx_chip_train_per_view_pad_cfg,
    va_nmx_chip_train_per_view_pad_fastwam_cfg,
    va_nmx_fold_towel_train_fastwam_cfg,
)

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'nmx_chip_train': va_nmx_chip_train_cfg,
    'nmx_chip_train_fastwam': va_nmx_chip_train_fastwam_cfg,
    'nmx_chip_train_mot': va_nmx_chip_train_mot_cfg,
    'nmx_fold_towel_train_mot_fastwam': va_nmx_fold_towel_train_mot_fastwam_cfg,
    'nmx_fold_towel_train_fastwam': va_nmx_fold_towel_train_fastwam_cfg,
    'nmx_chip_train_per_view_pad': va_nmx_chip_train_per_view_pad_cfg,
    'nmx_chip_train_per_view_pad_fastwam': va_nmx_chip_train_per_view_pad_fastwam_cfg,
    'nmx_chip_episode109_overfit': va_nmx_chip_episode109_overfit_cfg,
}
