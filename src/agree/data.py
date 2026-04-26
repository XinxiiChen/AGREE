import os
from pathlib import Path

import numpy as np
import scipy.io as scio
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[3]
GRAPH_DATA_DIR = Path(
    os.environ.get("AGREE_DATA_ROOT", REPO_ROOT / "data" / "graph_data")
)


def _resolve_graph_data_dir(graph_data_dir=None):
    if graph_data_dir is None:
        return GRAPH_DATA_DIR
    return Path(graph_data_dir)


def _ensure_dense(array_like):
    if sp.issparse(array_like):
        return array_like.toarray()
    return np.asarray(array_like)


def _load_npy_triplet(dataset_name: str, graph_data_dir=None):
    dataset_dir = _resolve_graph_data_dir(graph_data_dir) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    feat = np.load(dataset_dir / f"{dataset_name}_feat.npy", allow_pickle=True)
    label = np.load(dataset_dir / f"{dataset_name}_label.npy", allow_pickle=True)
    adj = np.load(dataset_dir / f"{dataset_name}_adj.npy", allow_pickle=True)

    feat = _ensure_dense(feat).astype(np.float32)
    adj = _ensure_dense(adj).astype(np.float32)
    label = np.asarray(label).reshape(-1)
    return feat, adj, label


def _load_exported_graph_dataset(dataset_name: str, graph_data_dir=None):
    dataset_dir = _resolve_graph_data_dir(graph_data_dir) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    feat_path = dataset_dir / f"{dataset_name}_feat.npy"
    label_path = dataset_dir / f"{dataset_name}_label.npy"
    adj_dense_path = dataset_dir / f"{dataset_name}_adj.npy"
    adj_sparse_path = dataset_dir / f"{dataset_name}_adj_sparse.npz"
    edge_index_path = dataset_dir / f"{dataset_name}_edge_index.npy"

    if not feat_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Exported dataset is incomplete under {dataset_dir}. "
            f"Expected at least {feat_path.name} and {label_path.name}."
        )

    feat = np.load(feat_path, allow_pickle=True)
    label = np.load(label_path, allow_pickle=True)

    if adj_dense_path.exists():
        adj = np.load(adj_dense_path, allow_pickle=True)
    elif adj_sparse_path.exists():
        adj = sp.load_npz(adj_sparse_path)
    elif edge_index_path.exists():
        edge_index = np.load(edge_index_path, allow_pickle=True)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Invalid edge_index shape in {edge_index_path}: {edge_index.shape}"
            )
        num_nodes = feat.shape[0]
        adj = sp.coo_matrix(
            (np.ones(edge_index.shape[1], dtype=np.float32), (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32,
        )
        adj = adj.tocsr()
        adj.data[:] = 1.0
        adj.eliminate_zeros()
    else:
        raise FileNotFoundError(
            f"No adjacency file found in {dataset_dir}. "
            f"Expected one of {adj_dense_path.name}, {adj_sparse_path.name}, or {edge_index_path.name}."
        )

    feat = _ensure_dense(feat).astype(np.float32)
    adj = _ensure_dense(adj).astype(np.float32)
    label = np.asarray(label).reshape(-1)
    return feat, adj, label


def _load_cora_raw(graph_data_dir=None):
    dataset_dir = _resolve_graph_data_dir(graph_data_dir) / "cora_quaternion_used"
    content_path = dataset_dir / "cora.content"
    cites_path = dataset_dir / "cora.cites"

    idx_features_labels = np.genfromtxt(content_path, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    _, _, labels = np.unique(
        idx_features_labels[:, -1],
        return_index=True,
        return_inverse=True
    )

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),
        dtype=np.int32
    ).reshape(edges_unordered.shape)
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32
    )
    adj = adj.T + adj
    adj = adj.minimum(1)
    return features.toarray(), adj.toarray(), labels


def _load_mat_dataset(file_name: str, graph_data_dir=None):
    mat_path = _resolve_graph_data_dir(graph_data_dir) / file_name
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT dataset not found: {mat_path}")

    data = scio.loadmat(mat_path)
    features = _ensure_dense(data["fea"]).astype(np.float32)
    adj = _ensure_dense(data["W"]).astype(np.float32)
    labels = np.asarray(data["gnd"]).reshape(-1)
    # Many MATLAB datasets are 1-based.
    if labels.min() == 1:
        labels = labels - 1
    return features, adj, labels


def load_data(name: str, graph_data_dir=None):
    name = name.lower()
    data_root = _resolve_graph_data_dir(graph_data_dir)

    npy_datasets = {
        "acm", "dblp", "citeseer", "bat", "uat",
        "cornell", "texas", "eat", "wisc", "film", "amap"
    }

    if name == "cora":
        return _load_cora_raw(data_root)
    if name in npy_datasets:
        return _load_npy_triplet(name, data_root)
    if name == "wiki":
        return _load_mat_dataset("wiki.mat", data_root)
    if name == "pubmed":
        pubmed_dir = data_root / "pubmed"
        exported_pubmed_files = [
            pubmed_dir / "pubmed_feat.npy",
            pubmed_dir / "pubmed_label.npy",
        ]
        if all(path.exists() for path in exported_pubmed_files):
            return _load_exported_graph_dataset("pubmed", data_root)
        return _load_mat_dataset("pubmed.mat", data_root)
    if name == "mgae_citeseer":
        return _load_mat_dataset("mgae_citeseer.mat", data_root)


    custom_dir = data_root / name
    if custom_dir.exists():
        return _load_exported_graph_dataset(name, data_root)

    raise ValueError(f"Unsupported dataset: {name}")
