import argparse
import sys
from .modules.model import HACA3

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Harmonization with HACA3.')
    parser.add_argument('--dataset-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--contrasts', type=str, nargs='+', required=True)
    parser.add_argument('--orientations', type=str, nargs='+', default=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--eta-dim', type=int, default=2)
    parser.add_argument('--pretrained-haca3', type=str, default=None)
    parser.add_argument('--pretrained-eta-encoder', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args(args)

    text_div = '=' * 10
    print(f'{text_div} BEGIN HACA3 TRAINING {text_div}')

    # ====== 1. INITIALIZE MODEL ======
    haca3 = HACA3(beta_dim=args.beta_dim, theta_dim=args.theta_dim, eta_dim=args.eta_dim,
                  pretrained_haca3=args.pretrained_haca3, pretrained_eta_encoder=args.pretrained_eta_encoder,
                  gpu_id=args.gpu_id)

    # ====== 2. LOAD DATASETS ======
    haca3.load_dataset(dataset_dirs=args.dataset_dirs, contrasts=args.contrasts, orientations=args.orientations,
                       batch_size=args.batch_size)

    # ====== 3. INITIALIZE TRAINING ======
    haca3.initialize_training(out_dir=args.out_dir, lr=args.lr)

    # ====== 4. BEGIN TRAINING ======
    haca3.train(epochs=args.epochs)

