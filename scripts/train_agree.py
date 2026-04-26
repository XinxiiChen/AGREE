import os
import sys
from pathlib import Path

def _normalize_thread_env() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(key)
        if value is None:
            os.environ[key] = "1"
            continue
        try:
            if int(value) <= 0:
                os.environ[key] = "1"
        except ValueError:
            os.environ[key] = "1"


_normalize_thread_env()

import torch
import warnings
import numpy as np
import argparse
import torch.nn.functional as F
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agree.config import CompleteConfig
from agree.data import load_data
from agree.logger import ExperimentLogger
from agree.models.agree import AGREE

warnings.filterwarnings('ignore')

def get_activation_function(name: str) -> callable:
    return getattr(F, name)

def build_layer_config(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> List[int]:
    """Build a layer configuration with fixed hidden width."""
    if n_layers < 2:
        raise ValueError("Number of layers must be at least 2")
        
    layers = [input_dim]
    for _ in range(n_layers - 1):
        layers.append(hidden_dim)
    layers.append(output_dim)
    return layers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='cora',
                      help='Dataset name. Built-in datasets are supported, and custom exported graph-data directories are also allowed.')
    parser.add_argument('--graph_data_root', type=str, default=None,
                      help='Optional root directory containing exported graph-data subsets, e.g. scaling_data.')
    parser.add_argument('--config_template', type=str, default=None,
                      help='Optional dataset config template to reuse for custom datasets, e.g. pubmed.')
    parser.add_argument('-n', '--n_layers', type=int, default=2,
                      choices=[2, 4, 6, 8, 10],
                      help='Number of layers')
    parser.add_argument('--runs_override', type=int, default=None,
                      help='Override config.common.runs for quick tests')
    parser.add_argument('--max_epoch_override', type=int, default=None,
                      help='Override dataset max_epoch for quick tests')
    parser.add_argument('--max_iter_override', type=int, default=None,
                      help='Override dataset max_iter for quick tests')
    parser.add_argument('--pre_iter_override', type=int, default=None,
                      help='Override dataset pre_iter for quick tests')
    parser.add_argument('--eval_every', type=int, default=5,
                      help='Evaluation interval during training')
    parser.add_argument('--skip_initial_eval', action='store_true',
                      help='Skip the initial clustering before training')
    parser.add_argument('--final_eval_only', action='store_true',
                      help='Disable intermediate evaluations and only evaluate at the end')
    parser.add_argument('--no_log', action='store_true',
                      help='Disable JSON result logging for quick smoke tests')
    args = parser.parse_args()
    
    logger = None
    if not args.no_log:
        logger = ExperimentLogger(
            "agree_runs",
            output_path=str(PROJECT_ROOT / "results")
        )
    
    config = CompleteConfig.from_yaml(str(PROJECT_ROOT / "configs" / "config.yaml"))

    seed = config.common.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.common.gpu

    name = args.dataset
    dataset_config_name = args.config_template if args.config_template is not None else name
    if not hasattr(config, dataset_config_name):
        raise ValueError(f"No dataset config named {dataset_config_name} in config.yaml")
    dataset_config = getattr(config, dataset_config_name)
    if args.runs_override is not None:
        config.common.runs = args.runs_override
    if args.max_epoch_override is not None:
        dataset_config.max_epoch = args.max_epoch_override
    if args.max_iter_override is not None:
        dataset_config.max_iter = args.max_iter_override
    if args.pre_iter_override is not None:
        dataset_config.pre_iter = args.pre_iter_override
    
    features, adjacency, labels = load_data(name, graph_data_dir=args.graph_data_root)

    input_dim = dataset_config.layers[0]
    hidden_dim = dataset_config.layers[1]
    output_dim = dataset_config.layers[2]
    layers = build_layer_config(input_dim, hidden_dim, output_dim, args.n_layers)
    print(layers)
    acts = [get_activation_function(dataset_config.acts[0])] * len(layers)

    acc_list, nmi_list, ari_list, f1_list = [], [], [], []

    for run_idx in range(config.common.runs):
        gae = AGREE(
            name, features, adjacency, labels,
            layers=layers,
            acts=acts,
            max_epoch=dataset_config.max_epoch,
            max_iter=dataset_config.max_iter,
            coeff_reg=dataset_config.coeff_reg,
            learning_rate=dataset_config.learning_rate,
            seed=seed,
            lam=np.power(2.0, dataset_config.lamSC),
            eval_every=args.eval_every,
            skip_initial_eval=args.skip_initial_eval,
            final_eval_only=args.final_eval_only
        )
        if torch.cuda.is_available():
            gae.cuda() 
        else:
            gae.cpu()
            
        gae.pretrain(dataset_config.pre_iter, learning_rate=dataset_config.pretrain_learning_rate)
        acc, nmi, ari, f1 = gae.run()

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        f1_list.append(f1)

        run_metrics = {
            "acc": float(acc),
            "nmi": float(nmi),
            "ari": float(ari),
            "f1": float(f1)
        }
        if logger is not None:
            logger.add_run_result(
                name, 
                layers, 
                run_metrics,
                max_epoch=dataset_config.max_epoch,
                repeater_runs=config.common.runs
            )
        gae = None
        
    if logger is not None:
        logger.save_results()

    print("\n")
    acc_list = np.array(acc_list)
    nmi_list = np.array(nmi_list)
    ari_list = np.array(ari_list)
    f1_list = np.array(f1_list)

    print(acc_list.mean(), "±", acc_list.std())
    print(nmi_list.mean(), "±", nmi_list.std())
    print(ari_list.mean(), "±", ari_list.std())
    print(f1_list.mean(), "±", f1_list.std())
