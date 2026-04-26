from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def move_column_to_end(arr: np.ndarray, col_index: list[int]) -> np.ndarray:
    moved = [arr[:, i] for i in col_index]
    arr = np.delete(arr, col_index, axis=1)
    return np.insert(arr, arr.shape[1], np.array(moved), axis=1)


def _factorize_labels(series: pd.Series) -> np.ndarray:
    labels, _ = pd.factorize(series)
    return labels.astype(np.int64)


@dataclass(frozen=True)
class MixedDatasetSpec:
    name: str
    no_nom_att: int
    no_ord_att: int
    no_num_att: int
    loader: Callable[[Path], tuple[np.ndarray, np.ndarray]]


def _load_zoo(root: Path) -> tuple[np.ndarray, np.ndarray]:
    replace = {"2": "a", "4": "b", "5": "c", "6": "d", "8": "e"}
    data = pd.read_csv(root / "zoo" / "zoo.data", header=None)
    features = data.iloc[:, 1:-1].copy()
    labels = _factorize_labels(data.iloc[:, -1])
    for key, value in replace.items():
        features[13].replace(key, value, inplace=True)
    return np.array(features, dtype=np.float32), labels


def _load_car(root: Path) -> tuple[np.ndarray, np.ndarray]:
    replace = {"vhigh": 3, "high": 2, "med": 1, "low": 0, "small": 0, "big": 2}
    data = pd.read_csv(root / "car_evaluation" / "car.data", header=None)
    features = data.iloc[:, :6].copy()
    labels = _factorize_labels(data.iloc[:, 6])
    for key, value in replace.items():
        features.replace(key, value, inplace=True)
    feature_array = move_column_to_end(features.values.astype(np.float32), [0, 1])
    return feature_array, labels


def _load_iris(root: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.genfromtxt(root / "iris" / "iris.data", dtype=np.dtype(str))
    features = []
    labels = []
    for item in raw:
        parts = item.split(",")
        features.append([float(value) for value in parts[:-1]])
        labels.append(parts[-1])
    features = np.array(features, dtype=np.float32)
    _, _, labels = np.unique(np.array(labels), return_index=True, return_inverse=True)
    return features, labels.astype(np.int64)


def _load_wine(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "wine" / "wine.data", header=None)
    labels = data.iloc[:, 0].values.astype(int) - 1
    features = data.iloc[:, 1:].values.astype(np.float32)
    return features, labels.astype(np.int64)


def _load_heart(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "heart_failure" / "heart.csv")
    labels = data.iloc[:, -1].values.astype(int)
    features = data.iloc[:, :-1].values.astype(np.float32)
    features = move_column_to_end(features, [0, 2, 4, 6, 7, 8, 11])
    return features, labels.astype(np.int64)


def _load_ttt(root: Path) -> tuple[np.ndarray, np.ndarray]:
    replace = {"x": 0, "o": 1, "b": 2}
    data = pd.read_csv(root / "tic_tac_toe" / "tic_tac_toe.data", header=None)
    features = data.iloc[:, :-1].copy()
    labels = _factorize_labels(data.iloc[:, -1])
    for key, value in replace.items():
        features.replace(key, value, inplace=True)
    return features.values.astype(np.float32), labels


def _load_lymphography(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "lymphography" / "lymphography.data", header=None)
    labels = data.iloc[:, 0].values.astype(int) - 1
    features = data.iloc[:, 1:].values.astype(np.float32)
    features = move_column_to_end(features, [8, 9, 17])
    features[:, :15] -= 1
    return features, labels.astype(np.int64)


def _load_yeast(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.DataFrame(np.genfromtxt(root / "yeast" / "yeast.data", dtype=np.dtype(str)))
    features = data.iloc[:, 1:9].values.astype(np.float32)
    labels = _factorize_labels(data.iloc[:, 9])
    return features, labels


def _load_breast(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "BC.csv", header=None)
    features = data.iloc[:, :-1].values.astype(np.float32) - 1
    labels = data.iloc[:, -1].values.astype(int) - 1
    return features, labels.astype(np.int64)


def _load_hayes(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "hayes_roth" / "hayes-roth.data", header=None)
    labels = data.iloc[:, -1].values.astype(int) - 1
    features = data.iloc[:, 1:-1].values.astype(np.float32)
    features = move_column_to_end(features, [1, 2])
    features[:, 0:2] -= 1
    return features, labels.astype(np.int64)


def _load_glass(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "glass_identification" / "glass.data", header=None)
    labels = data.iloc[:, -1].values.astype(int) - 1
    labels[163:] -= 1
    features = data.iloc[:, 1:-1].values.astype(np.float32)
    return features, labels.astype(np.int64)


def _load_aa(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "Autism-Adolescent" / "AA.csv", header=None)
    features = data.iloc[:, :-1].values.astype(np.float32) - 1
    labels = data.iloc[:, -1].values.astype(int) - 1
    return features, labels.astype(np.int64)


def _load_mm(root: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(root / "Mammographic" / "mammographic.csv", header=None)
    labels = data.iloc[:, -1].values.astype(int) - 1
    features = pd.get_dummies(data.iloc[:, :-1]).values.astype(np.float32)
    features[:, :-1] -= 1
    return features, labels.astype(np.int64)


DATASET_SPECS: dict[str, MixedDatasetSpec] = {
    "zoo": MixedDatasetSpec("zoo", 16, 0, 0, _load_zoo),
    "car": MixedDatasetSpec("car", 2, 3, 0, _load_car),
    "iris": MixedDatasetSpec("iris", 0, 0, 4, _load_iris),
    "wine": MixedDatasetSpec("wine", 0, 0, 13, _load_wine),
    "heart": MixedDatasetSpec("heart", 5, 0, 7, _load_heart),
    "ttt": MixedDatasetSpec("ttt", 9, 0, 0, _load_ttt),
    "lymphography": MixedDatasetSpec("lymphography", 15, 0, 3, _load_lymphography),
    "yeast": MixedDatasetSpec("yeast", 0, 0, 8, _load_yeast),
    "breast": MixedDatasetSpec("breast", 5, 4, 0, _load_breast),
    "hayes": MixedDatasetSpec("hayes", 4, 0, 0, _load_hayes),
    "glass": MixedDatasetSpec("glass", 0, 0, 9, _load_glass),
    "aa": MixedDatasetSpec("aa", 7, 0, 2, _load_aa),
    "mm": MixedDatasetSpec("mm", 4, 0, 1, _load_mm),
}


class MixedGraphPreprocessor:
    def __init__(
        self,
        x: np.ndarray,
        label: np.ndarray,
        no_nom_att: int,
        no_ord_att: int,
        no_num_att: int,
        construct_name: str = "IPD",
        adj_name: str = "dis",
    ) -> None:
        self.x = x.copy()
        self.label = label
        self.no_nom_att = no_nom_att
        self.no_ord_att = no_ord_att
        self.no_num_att = no_num_att
        self.n = x.shape[0]
        self.d = x.shape[1]
        self.no_values = [len(np.unique(x[:, t])) for t in range(self.d)]
        for i in range(self.d - self.no_num_att, self.d):
            col = self.x[:, i]
            denom = np.max(col) - np.min(col)
            self.x[:, i] = 0.0 if denom == 0 else (col - np.min(col)) / denom

        self.intra_pd = None
        self.cpd = None
        self.dis_matrix = None
        self.adjacent_matrix = None
        self.cw = None
        self.pbr_dis_matrix = None
        self.x_coded = None
        self.expanded_x = None

        self.bd_computer()
        self.pbr()

        if construct_name == "IPD":
            self.feature_ipd()
        elif construct_name == "IP":
            self.feature_ip()
        elif construct_name == "I":
            self.feature_i()
        else:
            raise ValueError(f"Unsupported construct_name: {construct_name}")

        if adj_name == "graph":
            self.graph_based_dissimilarity()
        elif adj_name == "dis":
            self.represent_graph()
        else:
            raise ValueError(f"Unsupported adj_name: {adj_name}")

    def bd_computer(self) -> None:
        x = self.x.copy()
        ia_pd_list = [None for _ in range(0, self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            all_sum = len(x[:, t])
            ia_pd = np.zeros((1, self.no_values[t]))
            for m in range(0, self.no_values[t]):
                locate_x_tm = x[:, t] == m
                ia_pd[0, m] = sum(locate_x_tm) / all_sum
            ia_pd_list[t] = ia_pd

        dis_matrix = [np.zeros((self.no_values[t], self.no_values[t])) for t in range(0, self.d - self.no_num_att)]
        cpd = [None for _ in range(self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            cpd[t] = [[None for _ in range(self.d)] for _ in range(self.no_values[t])]
            for m in range(0, self.no_values[t]):
                locate_x_tm = x[:, t] == m
                no_x_tm = sum(locate_x_tm)
                for r in range(0, self.d - self.no_num_att):
                    cpd[t][m][r] = np.zeros((1, self.no_values[r]))
                    for g in range(0, self.no_values[r]):
                        cpd[t][m][r][0, g] = sum(x[locate_x_tm, r] == g)
                    cpd[t][m][r] /= no_x_tm
                for r in range(self.d - self.no_num_att, self.d):
                    cpd[t][m][r] = np.zeros((1, 5))
                    for s in [0.0, 0.2, 0.4, 0.6]:
                        cpd[t][m][r][0, int(s * 5)] = np.sum((x[locate_x_tm, r] >= s - 0.0) * (x[locate_x_tm, r] < s))
                    cpd[t][m][r][0, 4] = no_x_tm - np.sum(cpd[t][m][r])
                    cpd[t][m][r] /= no_x_tm

        for t in range(0, self.no_nom_att):
            for m in range(0, self.no_values[t] - 1):
                for h in range(m + 1, self.no_values[t]):
                    cost_relate = np.zeros((1, self.d))
                    for r in range(0, self.no_nom_att):
                        diff_relate = cpd[t][h][r] - cpd[t][m][r]
                        cost_relate[0, r] = np.sum(np.abs(diff_relate)) / 2
                    for r in range(self.no_nom_att, self.d - self.no_num_att):
                        diff_relate = cpd[t][h][r] - cpd[t][m][r]
                        for s in range(0, self.no_values[r] - 1):
                            cost_relate[0, r] += np.abs(diff_relate[0, s])
                            diff_relate[0, s + 1] = diff_relate[0, s] + diff_relate[0, s + 1]
                        cost_relate[0, r] /= (self.no_values[r] - 1)
                    for r in range(self.d - self.no_num_att, self.d):
                        diff_relate = cpd[t][h][r] - cpd[t][m][r]
                        for s in range(0, 4):
                            cost_relate[0, r] += np.abs(diff_relate[0, s])
                            diff_relate[0, s + 1] = diff_relate[0, s] + diff_relate[0, s + 1]
                        cost_relate[0, r] /= 4
                    dis_matrix[t][m][h] = np.mean(cost_relate)
                    dis_matrix[t][h][m] = dis_matrix[t][m][h]

        for t in range(self.no_nom_att, self.d - self.no_num_att):
            dist_vct = np.zeros((1, self.no_values[t] - 1))
            for m in range(0, self.no_values[t] - 1):
                cost_relate = np.zeros((1, self.d))
                for r in range(0, self.no_nom_att):
                    diff_relate = cpd[t][m + 1][r] - cpd[t][m][r]
                    cost_relate[0, r] = np.sum(np.abs(diff_relate)) / 2
                for r in range(self.no_nom_att, self.d - self.no_num_att):
                    diff_relate = cpd[t][m + 1][r] - cpd[t][m][r]
                    for s in range(0, self.no_values[r] - 1):
                        cost_relate[0, r] += np.abs(diff_relate[0, s])
                        diff_relate[0, s + 1] += diff_relate[0, s]
                    cost_relate[0, r] /= (self.no_values[r] - 1)
                for r in range(self.d - self.no_num_att, self.d):
                    diff_relate = cpd[t][m + 1][r] - cpd[t][m][r]
                    for s in range(0, 4):
                        cost_relate[0, r] += np.abs(diff_relate[0, s])
                        diff_relate[0, s + 1] += diff_relate[0, s]
                    cost_relate[0, r] /= 4
                dist_vct[0, m] = np.mean(cost_relate)
            for m in range(0, self.no_values[t] - 1):
                for h in range(m + 1, self.no_values[t]):
                    dis_matrix[t][m][h] = np.sum(dist_vct[0, m:h])
                    dis_matrix[t][h][m] = dis_matrix[t][m][h]
            dis_matrix[t] /= np.max(dis_matrix[t])

        self.intra_pd = ia_pd_list
        self.cpd = cpd
        self.dis_matrix = dis_matrix

    def pbr(self) -> None:
        bd_mtx = self.dis_matrix.copy()
        x = self.x.copy()
        self.cw = np.zeros((1, self.d))
        for i in range(0, self.no_nom_att):
            self.cw[0, i] = np.max(bd_mtx[i])
        self.cw[0, self.no_nom_att:-1] = 1
        pbr_list = (np.arange(0, self.no_nom_att) + 1) * (np.array(self.no_values[0:self.no_nom_att]) > 2)
        pbr_list = (pbr_list[pbr_list != 0] - 1).tolist()
        npbr_list = np.setdiff1d(np.arange(0, self.d - self.no_num_att), pbr_list).tolist()
        num_pbr_att = len(pbr_list)
        pbr_dis = [None] * num_pbr_att
        for r in range(0, num_pbr_att):
            num_att_val = self.no_values[pbr_list[r]]
            cn2 = ((num_att_val * (num_att_val - 1)) // 2)
            pbr_dis[r] = [np.zeros((num_att_val, num_att_val)) for _ in range(cn2)]
            num_new_att = -1
            for v1 in range(0, self.no_values[pbr_list[r]] - 1):
                for v2 in range(v1 + 1, self.no_values[pbr_list[r]]):
                    num_new_att += 1
                    d12 = bd_mtx[pbr_list[r]][v1, v2]
                    plist = np.setdiff1d(np.arange(0, self.no_values[pbr_list[r]]), [v1, v2])
                    pval = np.zeros((1, self.no_values[pbr_list[r]]))
                    pval[0, v2] = d12
                    for vm in plist:
                        d1m = bd_mtx[int(pbr_list[r])][v1, vm]
                        d2m = bd_mtx[int(pbr_list[r])][v2, vm]
                        if d1m > d2m:
                            e = (d1m ** 2 - d2m ** 2 + d12 ** 2) / (2 * d12)
                            pval[0, vm] = e
                        else:
                            e = (d2m ** 2 - d1m ** 2 + d12 ** 2) / (2 * d12)
                            pval[0, vm] = d12 - e
                    pval -= np.min(pval)
                    for vv1 in range(0, self.no_values[pbr_list[r]] - 1):
                        for vv2 in range(vv1 + 1, self.no_values[pbr_list[r]]):
                            pbr_dis[r][num_new_att][vv1, vv2] = np.abs(pval[0, vv1] - pval[0, vv2])
                            pbr_dis[r][num_new_att][vv2, vv1] = pbr_dis[r][num_new_att][vv1, vv2]

        x_pbr = np.zeros((self.n, 0))
        dis_mtx_pbr = []
        for r in range(0, num_pbr_att):
            num_new_att = self.no_values[pbr_list[r]] * (self.no_values[pbr_list[r]] - 1) // 2
            expand_x = np.tile(x[:, pbr_list[r]:pbr_list[r] + 1], (1, int(num_new_att)))
            x_pbr = np.hstack((x_pbr, expand_x))
            dis_mtx_pbr.append(pbr_dis[r])

        num_list = [i for i in range(self.d - self.no_num_att, self.d)]
        column = npbr_list + num_list
        x_ncoded = x[:, column]
        self.x_coded = np.hstack((x_pbr, x_ncoded))

        dis_mtx = [None] * (self.d - self.no_num_att)
        for j in range(len(dis_mtx_pbr)):
            dis_mtx[pbr_list[j]] = dis_mtx_pbr[j]
        for j in range(len(npbr_list)):
            dis_mtx[npbr_list[j]] = bd_mtx[npbr_list[j]]
        self.pbr_dis_matrix = dis_mtx

    def graph_based_dissimilarity(self) -> None:
        x = self.x.copy()
        numerical_values = []
        for i in range(self.d - self.no_num_att, self.d):
            col = x[:, i]
            denom = np.max(col) - np.min(col)
            x[:, i] = 0.0 if denom == 0 else (col - np.min(col)) / denom
        for t in range(self.d - self.no_num_att, self.d):
            numerical_values.append(np.unique(x[:, t]))

        cpd = [None for _ in range(self.d - self.no_num_att)]
        for t in range(0, self.d - self.no_num_att):
            cpd[t] = [[None for _ in range(self.d)] for _ in range(self.no_values[t])]
            for m in range(0, self.no_values[t]):
                locate_x_tm = x[:, t] == m
                no_x_tm = sum(locate_x_tm)
                for r in range(0, self.d - self.no_num_att):
                    cpd[t][m][r] = np.zeros((1, self.no_values[r]))
                    for g in range(0, self.no_values[r]):
                        cpd[t][m][r][0, g] = sum(x[locate_x_tm, r] == g)
                    cpd[t][m][r] /= no_x_tm
                for r in range(self.d - self.no_num_att, self.d):
                    cpd[t][m][r] = np.zeros((1, self.no_values[r]))
                    for g in range(0, self.no_values[r]):
                        a = x[locate_x_tm, r]
                        b = numerical_values[r - self.no_nom_att - self.no_ord_att][g]
                        cpd[t][m][r][0, g] = sum(a == b)
                    cpd[t][m][r] /= no_x_tm

        diff_matrix = [[np.zeros((self.no_values[t], self.no_values[t])) for _ in range(self.d)] for t in range(0, self.d - self.no_num_att)]
        for d in range(0, self.no_nom_att):
            for r in range(0, self.d):
                for m in range(0, self.no_values[d] - 1):
                    for h in range(m + 1, self.no_values[d]):
                        diff_relate = cpd[d][m][r] - cpd[d][h][r]
                        if 0 <= r < self.no_nom_att:
                            phi = np.abs(np.sum(diff_relate[np.where(diff_relate > 0)]))
                        else:
                            max_tag = np.where(diff_relate == np.max(diff_relate))[1].tolist()
                            if self.no_nom_att <= r < self.d - self.no_num_att:
                                t_vector = np.linspace(0, 1, self.no_values[r])
                            else:
                                t_vector = np.array(numerical_values[r - self.no_nom_att - self.no_ord_att])
                            min_phi = float("inf")
                            for tag in max_tag:
                                t = np.abs(t_vector - t_vector[tag])
                                phi = np.sum(np.abs(diff_relate) * t)
                                if phi < min_phi:
                                    min_phi = phi
                            phi = min_phi
                        diff_matrix[d][r][m][h] = phi
                        diff_matrix[d][r][h][m] = phi

        for d in range(self.no_nom_att, self.d - self.no_num_att):
            for r in range(0, self.d):
                for m in range(0, self.no_values[d] - 1):
                    for h in range(m + 1, self.no_values[d]):
                        for v_mh in range(m, h):
                            diff_relate = cpd[d][v_mh][r] - cpd[d][v_mh + 1][r]
                            if 0 <= r < self.no_nom_att:
                                phi = np.abs(np.sum(diff_relate[np.where(diff_relate > 0)]))
                            else:
                                max_tag = np.where(diff_relate == np.max(diff_relate))[1].tolist()
                                if self.no_nom_att <= r < self.d - self.no_num_att:
                                    t_vector = np.linspace(0, 1, self.no_values[r])
                                else:
                                    t_vector = np.array(numerical_values[r - self.no_nom_att - self.no_ord_att])
                                min_phi = float("inf")
                                for tag in max_tag:
                                    t = np.abs(t_vector - t_vector[tag])
                                    phi = np.sum(np.abs(diff_relate) * t)
                                    if phi < min_phi:
                                        min_phi = phi
                                phi = min_phi
                            diff_matrix[d][r][m][h] += phi
                            diff_matrix[d][r][h][m] = diff_matrix[d][r][m][h]

        w_list = [[None for _ in range(self.d)] for _ in range(0, self.d - self.no_num_att)]
        for d in range(self.d - self.no_num_att):
            if d < self.no_nom_att:
                for r in range(0, self.d):
                    w_list[d][r] = (np.sum(diff_matrix[d][r]) / 2) / (self.no_values[d] * (self.no_values[d] - 1) / 2)
            else:
                for r in range(0, self.d):
                    sum_diff = 0
                    for i in range(0, self.no_values[d] - 1):
                        sum_diff += diff_matrix[d][r][i, i + 1]
                    w_list[d][r] = sum_diff / (self.no_values[d] - 1)

        dis_matrix = [np.zeros((self.no_values[t], self.no_values[t])) for t in range(0, self.d - self.no_num_att)]
        for d in range(0, self.d - self.no_num_att):
            for s in range(0, self.d):
                dis_matrix[d] += w_list[d][s] * diff_matrix[d][s]
        for d in range(0, self.d - self.no_num_att):
            if d < self.no_nom_att:
                dis_matrix[d] /= self.d
            else:
                dis_matrix[d] /= np.max(dis_matrix[d])
        connect_matrix = self._pairwise_distance_matrix(x, dis_matrix)
        self.adjacent_matrix = connect_matrix
        self.dis_matrix = dis_matrix

    def represent_graph(self) -> None:
        dis_matrix = self.dis_matrix.copy()
        x = self.x.copy()
        for i in range(self.d - self.no_num_att, self.d):
            col = x[:, i]
            denom = np.max(col) - np.min(col)
            x[:, i] = 0.0 if denom == 0 else (col - np.min(col)) / denom
        self.adjacent_matrix = self._pairwise_distance_matrix(x, dis_matrix)

    def _pairwise_distance_matrix(self, x: np.ndarray, dis_matrix) -> np.ndarray:
        connect_matrix = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(0, self.n):
            for j in range(i, self.n):
                diff_vector = np.zeros((1, self.d))
                for r in range(0, self.d - self.no_num_att):
                    ai = int(x[i, r])
                    aj = int(x[j, r])
                    diff_vector[0, r] = dis_matrix[r][ai, aj]
                for r in range(self.d - self.no_num_att, self.d):
                    diff_vector[0, r] = x[j, r] - x[i, r]
                distance = np.linalg.norm(diff_vector, ord=2)
                connect_matrix[i, j] = distance
                connect_matrix[j, i] = distance
        return connect_matrix

    def feature_ipd(self) -> None:
        intra_pd = self.intra_pd.copy()
        cpd = self.cpd.copy()
        pbr_dis_matrix = self.pbr_dis_matrix.copy()
        x = self.x.copy()
        feature_matrix = []
        for i in range(self.d - self.no_num_att, self.d):
            col = x[:, i]
            denom = np.max(col) - np.min(col)
            x[:, i] = 0.0 if denom == 0 else (col - np.min(col)) / denom
        for t in range(0, self.n):
            expand_feature_vector = np.zeros((1, 0))
            for r in range(0, self.d - self.no_num_att):
                values = int(x[t, r])
                expand_feature_vector = np.hstack((expand_feature_vector, intra_pd[r][0:1, values:values + 1]))
                for i in range(0, self.d - self.no_num_att):
                    expand_feature_vector = np.hstack((expand_feature_vector, cpd[r][values][i]))
                if r < (self.d - self.no_num_att - self.no_ord_att):
                    if isinstance(pbr_dis_matrix[r], list):
                        for j in range(0, len(pbr_dis_matrix[r])):
                            expand_feature_vector = np.hstack((expand_feature_vector, pbr_dis_matrix[r][j][values:values + 1, :]))
                    else:
                        expand_feature_vector = np.hstack((expand_feature_vector, pbr_dis_matrix[r][values:values + 1, :]))
                else:
                    expand_feature_vector = np.hstack((expand_feature_vector, pbr_dis_matrix[r][values:values + 1, :]))
            for r in range(self.d - self.no_num_att, self.d):
                expand_feature_vector = np.hstack((expand_feature_vector, x[t:t + 1, r:r + 1]))
            feature_matrix.append(expand_feature_vector[0, :])
        self.expanded_x = np.array(feature_matrix, dtype=np.float32)

    def feature_ip(self) -> None:
        intra_pd = self.intra_pd.copy()
        cpd = self.cpd.copy()
        x = self.x.copy()
        feature_matrix = []
        for i in range(self.d - self.no_num_att, self.d):
            col = x[:, i]
            denom = np.max(col) - np.min(col)
            x[:, i] = 0.0 if denom == 0 else (col - np.min(col)) / denom
        for t in range(0, self.n):
            expand_feature_vector = np.zeros((1, 0))
            for r in range(0, self.d - self.no_num_att):
                values = int(x[t, r])
                expand_feature_vector = np.hstack((expand_feature_vector, intra_pd[r][0:1, values:values + 1]))
                for i in range(0, self.d - self.no_num_att):
                    expand_feature_vector = np.hstack((expand_feature_vector, cpd[r][values][i]))
            for r in range(self.d - self.no_num_att, self.d):
                expand_feature_vector = np.hstack((expand_feature_vector, x[t:t + 1, r:r + 1]))
            feature_matrix.append(expand_feature_vector[0, :])
        self.expanded_x = np.array(feature_matrix, dtype=np.float32)

    def feature_i(self) -> None:
        intra_pd = self.intra_pd.copy()
        x = self.x.copy()
        feature_matrix = []
        for i in range(self.d - self.no_num_att, self.d):
            col = x[:, i]
            denom = np.max(col) - np.min(col)
            x[:, i] = 0.0 if denom == 0 else (col - np.min(col)) / denom
        for t in range(0, self.n):
            expand_feature_vector = np.zeros((1, 0))
            for r in range(0, self.d - self.no_num_att):
                values = int(x[t, r])
                expand_feature_vector = np.hstack((expand_feature_vector, intra_pd[r][0:1, values:values + 1]))
            for r in range(self.d - self.no_num_att, self.d):
                expand_feature_vector = np.hstack((expand_feature_vector, x[t:t + 1, r:r + 1]))
            feature_matrix.append(expand_feature_vector[0, :])
        self.expanded_x = np.array(feature_matrix, dtype=np.float32)


def load_mixed_raw_dataset(name: str, raw_data_root: str | Path) -> tuple[np.ndarray, np.ndarray, MixedDatasetSpec]:
    dataset_name = name.lower()
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported mixed dataset: {name}")
    spec = DATASET_SPECS[dataset_name]
    features, labels = spec.loader(Path(raw_data_root))
    return features.astype(np.float32), labels.astype(np.int64), spec


def build_mixed_graph_dataset(
    name: str,
    raw_data_root: str | Path,
    construct_name: str = "IPD",
    adj_name: str = "dis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, spec = load_mixed_raw_dataset(name, raw_data_root)
    preprocessor = MixedGraphPreprocessor(
        features,
        labels,
        no_nom_att=spec.no_nom_att,
        no_ord_att=spec.no_ord_att,
        no_num_att=spec.no_num_att,
        construct_name=construct_name,
        adj_name=adj_name,
    )
    return preprocessor.expanded_x, preprocessor.adjacent_matrix.astype(np.float32), labels
