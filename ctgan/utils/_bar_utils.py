"""Module containing progress bar utilities.

They were implemented by
`Drew Szurko <https://github.com/drewszurko/tensorflow-WGAN-GP>`.
"""
import shutil
from tqdm.autonotebook import tqdm
from functools import reduce


class ProgressBar(tqdm):
    """Custom progress bar, designed for CTGAN training.

    ProgressBar inherits from ``tqdm.tqdm`` to create an informative progress
    bar in each training epoch.

    Parameters
    ----------
    total_samples: int
        Number of total samples, i.e., dataset size.

    batch_size: int
        Size of the current training batch.

    epoch: int
        Current training epoch.

    num_epochs: int
        Total number of training epochs.

    metrics: list, or dict
        List or dictionary of which metrics to display.

    """

    @classmethod
    def _get_terminal_width(cls):
        """Method that returns the execution terminal width.

        Returns
        -------
        int
            The execution terminal width if possible, otherwise, 120.

        """
        width = shutil.get_terminal_size(fallback=(200, 24))[0]
        return width if width != 0 else 120

    def __init__(self, total_samples, batch_size, epoch, num_epochs, metrics):
        postfix = {m: f'{0:6.3f}' for m in metrics}
        postfix[1] = 1
        str_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  ' \
                     'ETA: {remaining}  Elapsed Time: {elapsed}  ' + \
                     reduce(lambda x, y: x + y,
                            ["%s:{postfix[%s]}  " % (m, m) for m in metrics],
                            "")
        super(ProgressBar, self).__init__(
            total=(total_samples // batch_size) * batch_size,
            ncols=int(ProgressBar._get_terminal_width() * .9),
            desc=tqdm.write(f'Epoch {epoch + 1}/{num_epochs}'),
            postfix=postfix,
            bar_format=str_format,
            unit='samples',
            miniters=10)
        self._batch_size = batch_size

    def update(self, metrics):
        """Updates the progress bar metrics.

        Parameters
        ----------
        metrics: dict[str, tf.Metrics]
            Dictionary mapping the metrics to their current values.

        """
        for m in metrics:
            self.postfix[m] = f'{metrics[m].result():6.3f}'
        super(ProgressBar, self).update(self._batch_size)

    def close(self):
        """
        Closes the current progress bar.
        """
        super(ProgressBar, self).close()
