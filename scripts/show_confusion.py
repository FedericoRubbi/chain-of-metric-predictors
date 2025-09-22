import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--labels', type=str, default=None, help='Comma-separated class names (optional)')
    parser.add_argument('--normalize', action='store_true', help='Row-normalize the confusion matrix')
    args = parser.parse_args()

    cm_path = os.path.join(args.run_dir, 'confusion.npy')
    if not os.path.exists(cm_path):
        print(f"Not found: {cm_path}. Run eval.py --confusion first.")
        return

    cm = np.load(cm_path)
    if args.normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    labels = None
    if args.labels is not None:
        labels = args.labels.split(',')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix')
    plt.colorbar()
    if labels is not None and len(labels) == cm.shape[0]:
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    out = os.path.join(args.run_dir, 'confusion.png')
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
