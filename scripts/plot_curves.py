import argparse
import json
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving only')
    args = parser.parse_args()

    log_path = os.path.join(args.run_dir, 'log.jsonl')
    if not os.path.exists(log_path):
        print(f"No log.jsonl in {args.run_dir}")
        return

    train_steps = []
    train_loss = []
    train_acc_last = []
    val_epochs = []
    val_acc_last = []

    with open(log_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('phase') == 'train':
                train_steps.append(rec.get('step', len(train_steps)+1))
                if 'loss' in rec and rec['loss'] is not None:
                    train_loss.append(rec['loss'])
                if 'acc_last' in rec and rec['acc_last'] is not None:
                    train_acc_last.append(rec['acc_last'])
            elif rec.get('phase') == 'val':
                val_epochs.append(rec.get('epoch'))
                if 'acc_last' in rec and rec['acc_last'] is not None:
                    val_acc_last.append(rec['acc_last'])
                elif 'acc' in rec and rec['acc'] is not None:
                    val_acc_last.append(rec['acc'])

    # Plot training loss
    if train_steps and train_loss:
        plt.figure()
        plt.plot(train_steps[:len(train_loss)], train_loss)
        plt.xlabel('Step')
        plt.ylabel('Train Loss')
        plt.title('Training Loss')
        out = os.path.join(args.run_dir, 'plot_train_loss.png')
        plt.savefig(out, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
        print(f"Saved {out}")

    # Plot train acc (last layer)
    if train_steps and train_acc_last:
        plt.figure()
        plt.plot(train_steps[:len(train_acc_last)], train_acc_last)
        plt.xlabel('Step')
        plt.ylabel('Train Acc (last layer)')
        plt.title('Training Accuracy (last layer)')
        out = os.path.join(args.run_dir, 'plot_train_acc_last.png')
        plt.savefig(out, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
        print(f"Saved {out}")

    # Plot val acc (last)
    if val_epochs and val_acc_last:
        plt.figure()
        plt.plot(val_epochs[:len(val_acc_last)], val_acc_last, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Val Acc (last)')
        plt.title('Validation Accuracy (last layer)')
        out = os.path.join(args.run_dir, 'plot_val_acc_last.png')
        plt.savefig(out, bbox_inches='tight')
        if args.show:
            plt.show()
        plt.close()
        print(f"Saved {out}")


if __name__ == '__main__':
    main()
