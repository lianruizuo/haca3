import sys
import argparse
from modules.fusion_model import FusionNet

def main(args=None):
    args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Unsupervised harmonization via disentanglement.')
    parser.add_argument('--dataset-dirs', nargs='+', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='../')
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(args)

    # initialize model
    trainer = FusionNet(pretrained_model=args.pretrained_model, gpu=args.gpu)

    trainer.load_dataset(dataset_dirs=args.dataset_dirs, batch_size=args.batch_size)

    trainer.initialize_training(out_dir=args.out_dir, lr=args.lr)

    trainer.train(epochs=args.epochs)

if __name__ == '__main__':
    main()
