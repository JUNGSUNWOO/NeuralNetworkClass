import sys
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch


def parse_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        value = value.lower()
        if value in ['true', 'ok', '1']:
            return True
    return False


def plot_dir(logdir, prefix):
    logfiles = list(logdir.glob('log_*'))

    if len(logfiles) == 0:
        return

    # collect log
    train_log, val_log = {}, {}
    for log_f in logfiles:
        log = torch.load(str(log_f))
        train_log.update(log['train'])
        val_log.update(log['val'])

    # prepair logs for plot
    train_step, train_loss, train_acc = [], [], []
    for k in sorted(train_log.keys()):
        train_step.append(k)
        data = train_log[k]
        train_loss.append(data['loss'])
        train_acc.append(data['acc'])

    val_step, val_loss, val_acc = [], [], []
    for k in sorted(val_log.keys()):
        val_step.append(k)
        data = val_log[k]
        val_loss.append(data['loss'])
        val_acc.append(data['acc'])

    # plot logs
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(train_step, train_loss, '-', label=prefix + 'train')
    plt.plot(val_step, val_loss, '-', label=prefix + 'val')
    plt.xlabel('step')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(train_step, train_acc, '-', label=prefix + 'train')
    plt.plot(val_step, val_acc, '-', label=prefix + 'val')
    plt.xlabel('step')
    plt.legend()


def main():
    log_path = './log_AGE_100'
    logdir = Path(log_path)
    dirs = [logdir]

    for d in dirs:
        subfix = d.name + '-'
        plot_dir(d, subfix)

    plt.show()


main()