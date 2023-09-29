import argparse
import sys
from .modules.model import HACA3


def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(
        description='Harmonization with attention based contrast, anatomy, and artifacts awareness')
    parser.add_argument('--dataset-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--contrasts', type=str, nargs='+', required=True)
    parser.add_argument('--orientations', type=str, nargs='+', default=['axial'])
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--pretrained-harmonization', type=str, default=None)
    parser.add_argument('--pretrained-eta-encoder', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(args)

    text_div = "=" * 10
    print(f"{text_div} BEGIN TRAINING {text_div}")

    # ==== INITIALIZE MODEL TRAINING ====
    haca3 = HACA3(
        beta_dim=5,
        theta_dim=2,
        eta_dim=2,
        pretrained_harmonization=args.pretrained_harmonization,
        pretrained_eta_encoder=args.pretrained_eta_encoder,
        gpu=args.gpu
    )

    # ==== LOAD DATASETS ====
    haca3.load_dataset(
        dataset_dirs=args.dataset_dirs,
        contrasts=args.contrasts,
        orientations=args.orientations,
        batch_size=args.batch_size
    )

    haca3.initialize_training(out_dir=args.out_dir, lr=args.lr)

    # ==== START TRAINING ====
    haca3.train_harmonization(epochs=args.epochs)
