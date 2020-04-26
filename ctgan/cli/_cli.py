import argparse
import pandas as pd

from ..synthesizer import CTGANSynthesizer
from ._load_demo import load_demo


def _parse_args():
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        The namespace containing the CLI arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-f', '--file_path', default=None, type=str,
                        help='File path to where a CTGAN Synthesizer model is stored.'
                             ' If this value is None, a new instance will be created.')
    parser.add_argument('-l', '--log_dir', default=None, type=str,
                        help='Directory to where log files will be stored.')
    parser.add_argument('-z', '--z_dim', default=128, type=int,
                        help='Embedding dimension of fake samples.')
    parser.add_argument('-p', '--pac', default=10, type=int,
                        help='Size of Pac framework.')
    parser.add_argument('-g', '--gen_dims', default=(256, 256),
                        help='Comma separated list of Generator layer sizes.')
    parser.add_argument('-c', '--crt_dims', default=(256, 256),
                        help='Comma separated list of Critic layer sizes.')
    parser.add_argument('-s', '--l2_scale', default=1e-6, type=float,
                        help='L2 regularization of the Generator optimizer.')
    parser.add_argument('-b', '--batch_size', default=500, type=int,
                        help='Training batch size.')
    parser.add_argument('-w', '--gp_lambda', default=10.0, type=float,
                        help='Gradient Penalty lambda.')
    parser.add_argument('-t', '--tau', default=10.0, type=float,
                        help='Gumbel-Softmax non-negative scalar temperature.')
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-d', '--discrete_columns',
                        help='Comma separated list of discrete columns,'
                             ' no whitespaces')
    parser.add_argument('-i', '--training_data', default='demo', type=str,
                        help='Path to training data')
    parser.add_argument('-o', '--output_model', default=None, type=str,
                        help='Path of the model output file')
    parser.add_argument('num_samples', type=int,
                        help='Number of rows to sample.')
    parser.add_argument('output_data', help='Path of the synthetic output file')
    return parser.parse_args()


def _parse_dims(dims):
    """Helper method to parse comma-separated NN layer dimension values.

    Parameters
    ----------
    dims: str
        Comma-separated string values. Example `256,256`

    Returns
    -------
    list[int]
        A list containing the NN layer dimensions values.
    """
    if isinstance(dims, str):
        return [int(d) for d in dims.split(',')]
    return dims


def _read_csv(csv_file, discrete=None):
    """Read CSV file and parse discrete columns.

    Parameters
    ----------
    csv_file: str
        File path to CSV file.

    discrete: str
        Comma separated string values. Example: `col1,col2`

    Returns
    -------
    tuple(pandas.DataFrame, list[str])
        A tuple containing the DataFrame of the input CSV file, and the chosen
        discrete columns.
    """
    if csv_file == 'demo':
        return load_demo()
    discrete_cols = discrete if discrete is not None else discrete.split(',')
    return pd.read_csv(csv_file), discrete_cols


def cli():
    """CTGAN command-line interface.

    The following list of parameters describes the inputs of CLI.

    Other Parameters
    ----------------
    file_path: str, default=None
        File path to where a CTGAN Synthesizer model is stored.
        If this value is None, a new instance will be created.

    log_dir: str, default=None
        Directory to where log files will be stored.

    z_dim': int, default=128
        Embedding dimension of fake samples.

    pac: int, default=10
        Size of Pac framework.

    gen_dims: str, default=(256, 256)
        Comma separated list of Generator layer sizes.

    crt_dims: str, default=(256, 256)
        Comma separated list of Critic layer sizes.

    l2_scale: float, default=1e-6
        L2 regularization of the Generator optimizer.

    batch_size: int, default=500
        Training batch size.

    gp_lambda: float, default=10.0
        Gradient Penalty lambda.

    tau: float, default=10.0
        Gumbel-Softmax non-negative scalar temperature.

    epochs: int, default=300
        Number of training epochs.

    discrete_columns: str, default=None
        Comma separated list of discrete columns, no whitespaces.

    training_data: str, default='demo'
        Path to training data. Defaults to loading the demo dataset.

    output_model: str, default=None
        Path of the model output file.

    num-samples: int
        Number of rows to sample.

    output_data: str
        Path of the synthetic output file

    Examples
    --------
    >>> from ctgan.cli import cli()
    >>> cli()
    usage:  [-h] [-f FILE_PATH] [-l LOG_DIR] [-z Z_DIM] [-p PAC] [-g GEN_DIMS]
            [-c CRT_DIMS] [-s L2_SCALE] [-b BATCH_SIZE] [-w GP_LAMBDA] [-t TAU]
            [-e EPOCHS] [-d DISCRETE_COLUMNS] [-i TRAINING_DATA] [-o OUTPUT_MODEL]
            num_samples output_data

    """
    args = _parse_args()
    args.gen_dims = _parse_dims(args.gen_dims)
    args.crt_dims = _parse_dims(args.crt_dims)

    data, discrete_columns = _read_csv(args.training_data, args.discrete_columns)

    model = CTGANSynthesizer(
        file_path=args.file_path,
        log_dir=args.log_dir,
        z_dim=args.z_dim,
        pac=args.pac,
        gen_dim=args.gen_dims,
        crt_dim=args.crt_dims,
        l2_scale=args.l2_scale,
        batch_size=args.batch_size,
        gp_lambda=args.gp_lambda,
        tau=args.tau)

    if not args.file_path:
        model.train(
            data,
            discrete_columns,
            args.epochs)

    num_samples = args.num_samples or len(data)
    sampled = model.sample(num_samples)
    sampled.to_csv(args.output_data, index=False)

    if args.output_model:
        model.dump(args.output_model, overwrite=True)

