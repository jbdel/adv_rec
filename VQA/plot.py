import matplotlib.pyplot as plt
import numpy as np

# x = ax.lines[0].get_xdata()
# y = ax.lines[0].get_ydata()
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_ylabel("Loss")
# ax.set_xlabel("Learning Rate")
# ax.set_xscale('log')
# plt.show(fig)

def range_of(x):
    "Create a range from 0 to `len(x)`."
    return list(range(len(x)))

def plot(recorder, skip_start: int = 10, skip_end: int = 5, suggestion: bool = False,
         **kwargs):
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
    lrs = recorder._split_list(recorder.lrs, skip_start, skip_end)
    losses = recorder._split_list(recorder.losses, skip_start, skip_end)
    losses = [x.item() for x in losses]
    if 'k' in kwargs: losses = recorder.smoothen_by_spline(lrs, losses, **kwargs)
    fig, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    if suggestion:
        try:
            mg = (np.gradient(np.array(losses))).argmin()
        except:
            print("Failed to compute the gradients, there might not be enough points.")
            return
        print(f"Min numerical gradient: {lrs[mg]:.2E}")
        ax.plot(lrs[mg], losses[mg], markersize=10, marker='o', color='red')
        recorder.min_grad_lr = lrs[mg]
    return fig, ax

def plot_losses(recorder, skip_start:int=0, skip_end:int=0):
    "Plot training and validation losses."
    fig, ax = plt.subplots(1,1)
    losses = recorder._split_list(recorder.losses, skip_start, skip_end)
    iterations = recorder._split_list(range_of(recorder.losses), skip_start, skip_end)
    ax.plot(iterations, losses, label='Train')
    val_iter = recorder._split_list_val(np.cumsum(recorder.nb_batches), skip_start, skip_end)
    val_losses = recorder._split_list_val(recorder.val_losses, skip_start, skip_end)
    ax.plot(val_iter, val_losses, label='Validation')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batches processed')
    ax.legend()
    return fig, ax

def plot_metrics(recorder, skip_start:int=0, skip_end:int=0):
    "Plot metrics collected during training."
    assert len(recorder.metrics) != 0, "There are no metrics to plot."
    fig, axes = plt.subplots(len(recorder.metrics[0]),1,figsize=(6, 4*len(recorder.metrics[0])))
    val_iter = recorder._split_list_val(np.cumsum(recorder.nb_batches), skip_start, skip_end)
    axes = axes.flatten() if len(recorder.metrics[0]) != 1 else [axes]
    for i, ax in enumerate(axes):
        values = [met[i] for met in recorder.metrics]
        values = recorder._split_list_val(values, skip_start, skip_end)
        ax.plot(val_iter, values)
        ax.set_ylabel(str(recorder.metrics_names[i]))
        ax.set_xlabel('Batches processed')
    return fig, axes


def plot_lr(recorder, show_moms=False, skip_start:int=0, skip_end:int=0):
    "Plot learning rate, `show_moms` to include momentum."
    lrs = recorder._split_list(recorder.lrs, skip_start, skip_end)
    iterations = recorder._split_list(range_of(recorder.lrs), skip_start, skip_end)
    if show_moms:
        moms = recorder._split_list(recorder.moms, skip_start, skip_end)
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].plot(iterations, lrs)
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('Learning Rate')
        ax[1].plot(iterations, moms)
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Momentum')
    else:
        fig, ax = plt.subplots()
        ax.plot(iterations, lrs)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Learning Rate')
    return fig, ax


