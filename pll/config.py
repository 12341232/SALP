
import argparse

def get_config():
    p = argparse.ArgumentParser(description="SSL-PLL (Dual-Student + EMA + CTG) with auto-download datasets")
    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "Fashion-MNIST"], help="Dataset")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--img_size", type=int, default=32)
    # Split: use labeled_rate (20% or 50%) as requested; if set, overrides labels_per_class
    p.add_argument("--labeled_rate", type=float, default=0.2, choices=[0.2, 0.5], help="Use 20% or 50% labeled rate")
    p.add_argument("--labels_per_class", type=int, default=None, help="If provided, overrides labeled_rate-derived value")
    p.add_argument("--unlabeled_ratio", type=float, default=7.0, help="unlabeled / labeled ratio (only used if not using labeled_rate exact computation)")
    # Partial label params
    p.add_argument("--pll_ambiguity_min", type=int, default=1, help="min size of candidate set (incl true label)")
    p.add_argument("--pll_ambiguity_max", type=int, default=3, help="max size of candidate set (incl true label)")
    # Optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup_epochs", type=int, default=5)
    # EMA
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA momentum")
    # Augment
    p.add_argument("--cutout", type=int, default=0, help="cutout size (0 = off)")
    p.add_argument("--use_mixup", action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.75)
    # CTG (Class Transition Graph / PRG-like module)
    p.add_argument("--ctg_window_batches", type=int, default=128, help="how many recent batches to track transitions")
    p.add_argument("--ctg_self_loop_alpha", type=float, default=0.10, help="self-loop strength in H diag")
    # Pseudo-label selection & class temperature scaling
    p.add_argument("--pl_quantile", type=float, default=0.5, help="quantile for accepting unlabeled pseudo labels (adaptive)")
    p.add_argument("--class_temp_beta", type=float, default=1.0, help="class temperature scaling beta")
    p.add_argument("--class_temp_gamma", type=float, default=0.5, help="class temperature scaling gamma")
    # Logging / ckpt
    p.add_argument("--out_dir", type=str, default="./runs/pll_ssl_best_v2")
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()
