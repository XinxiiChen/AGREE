import json
import math
import time
from sklearn.cluster import KMeans
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from agree.core_qnn.quaternion_ops import *
from agree.metrics import cal_clustering_metric



def to_tensor(X):
    if type(X) is torch.Tensor:
        return X
    return torch.Tensor(X)

def get_Laplacian(A):
    device = A.device
    dim = A.shape[0]
    L = A + torch.eye(dim).to(device)
    D = L.sum(dim=1)
    sqrt_D = D.pow(-1 / 2)
    Laplacian = sqrt_D * (sqrt_D * L).t()
    return Laplacian

def get_Laplacian_from_weights(weights):
    """Normalize a dense weight matrix in Laplacian form."""
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree



class QGNNLayer(Module):
    def __init__(self, in_features, out_features, quaternion_ff=True,
                 act=F.relu, init_criterion='he', weight_init='quaternion',
                 seed=None):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.quaternion_ff = quaternion_ff
        self.act = act

        if self.quaternion_ff:
            self.register_parameter('r', Parameter(torch.Tensor(self.in_features, self.out_features)))
            self.register_parameter('i', Parameter(torch.Tensor(self.in_features, self.out_features)))
            self.register_parameter('j', Parameter(torch.Tensor(self.in_features, self.out_features)))
            self.register_parameter('k', Parameter(torch.Tensor(self.in_features, self.out_features)))
        else:
            self.register_parameter('commonLinear', Parameter(torch.Tensor(self.in_features, self.out_features)))

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        if self.quaternion_ff:
            winit = {'quaternion': quaternion_init,
                     'unitary': unitary_init}[self.weight_init]
            affect_init(self.r, self.i, self.j, self.k, winit,
                        self.rng, self.init_criterion)

        else:
            stdv = math.sqrt(6.0 / (self.commonLinear.size(0) + self.commonLinear.size(1)))
            self.commonLinear.data.uniform_(-stdv, stdv)


    def forward(self, x, adj):
        if x.device != self.r.device:
            x = x.to(self.r.device)
        if adj.device != self.r.device:
            adj = adj.to(self.r.device)
            
        if self.quaternion_ff:
            r1 = torch.cat([self.r, -self.i, -self.j, -self.k], dim=0)
            i1 = torch.cat([self.i, self.r, -self.k, self.j], dim=0)
            j1 = torch.cat([self.j, self.k, self.r, -self.i], dim=0)
            k1 = torch.cat([self.k, -self.j, self.i, self.r], dim=0)
            self.hamilton_matrix = torch.cat([r1, i1, j1, k1], dim=1)
            out = torch.mm(adj, torch.mm(x, self.hamilton_matrix))
        else:
            out = torch.mm(adj, torch.mm(x, self.commonLinear))

        return self.act(out)


class GCGQ(Module):
    def __init__(self,
                 name,
                 X,
                 A,
                 labels,
                 layers=None,
                 acts=None,
                 max_epoch=10,
                 max_iter=50,
                 learning_rate=10 ** -2,
                 coeff_reg=10 ** -3,
                 seed=114514,
                 lam=-1,
                 eval_every=5,
                 skip_initial_eval=False,
                 final_eval_only=False,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                 ):
        super(GCGQ, self).__init__()
        self.name = name
        self.device = device
        self.X = to_tensor(X).to(self.device)
        self.adjacency = to_tensor(A).to(self.device)
        self.labels = to_tensor(labels).to(self.device)

        self.n_clusters = self.labels.unique().shape[0]
        if layers is None:
            layers = [32, 16]
        self.layers = layers
        self.acts = acts
        assert len(self.acts) == len(self.layers)
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.coeff_reg = coeff_reg
        self.seed = seed
        self.eval_every = eval_every
        self.skip_initial_eval = skip_initial_eval
        self.final_eval_only = final_eval_only
        self.pretrain_time_sec = 0.0
        self.train_time_sec = 0.0
        self.eval_time_sec = 0.0

        self.data_size = self.X.shape[0]
        self.input_dim = self.X.shape[1]

        self.indicator = self.X
        self.embedding = self.X
        self.best_embedding = None
        self.links = 0
        self.lam = lam
        self._build_up()

    def _build_up(self):
        self.linear = torch.nn.Linear(self.input_dim, self.layers[0] * 4).to(self.device)
        self.quaternion_module_list = nn.ModuleList()
        
        for i in range(len(self.layers) - 1):
            self.quaternion_module_list.append(QGNNLayer(self.layers[i] * 4, self.layers[i + 1] * 4, quaternion_ff=True, act=self.acts[i], init_criterion='he', weight_init='quaternion', seed=self.seed))

    def forward(self, Laplacian):
        input = self.linear(self.X)

        for i in range(len(self.quaternion_module_list)):
            input = self.quaternion_module_list[i](input, Laplacian)

        self.embedding = input.view(self.data_size, 4, self.layers[-1]).mean(dim=1)
        return self.embedding

    def _compute_regularization_loss(self):
        """L1 regularization for weights"""
        l1_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name or any(x in name for x in ['r', 'i', 'j', 'k']):
                l1_loss += torch.abs(param).sum()
        return l1_loss
    
    def _iter_loss_row_ranges(self):
        chunk_size = 1024
        for start in range(0, self.data_size, chunk_size):
            yield start, min(start + chunk_size, self.data_size)

    def _embedding_similarity_chunk(self, start, end, target_device=None):
        row_embedding = self.embedding[start:end]
        col_embedding = self.embedding
        if target_device is not None:
            row_embedding = row_embedding.to(target_device)
            col_embedding = col_embedding.to(target_device)
        return row_embedding.matmul(col_embedding.t())

    def _compute_structural_loss(self):
        """Structural consistency loss without materializing a dense Laplacian."""
        size = self.X.shape[0]
        embedding = self.embedding
        summed_embedding = embedding.sum(dim=0)
        degree = embedding.matmul(summed_embedding)
        squared_norm = embedding.pow(2).sum(dim=1)
        gram = embedding.t().matmul(embedding)
        return (torch.dot(degree, squared_norm) - gram.pow(2).sum()) / size
    
    def _compute_reconstruction_loss(self, zero_diagonal=False):
        """Calculate reconstruction loss chunk-by-chunk from embeddings."""
        epsilon = 1e-7
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss = self.embedding.new_zeros(())

        for start, end in self._iter_loss_row_ranges():
            recons_chunk = self._embedding_similarity_chunk(start, end)
            if zero_diagonal and start < end:
                local_diag = torch.arange(end - start, device=recons_chunk.device)
                global_diag = torch.arange(start, end, device=recons_chunk.device)
                recons_chunk = recons_chunk.clone()
                recons_chunk[local_diag, global_diag] = 0.0
            adj_chunk = self.adjacency[start:end]
            pos_log = torch.log(torch.clamp_min(recons_chunk, epsilon))
            neg_log = torch.log(torch.clamp_min(1.0 - recons_chunk, epsilon))
            loss = loss - (pos_weight * adj_chunk * pos_log + (1.0 - adj_chunk) * neg_log).sum()

        return loss / (self.data_size ** 2)
    
    def _build_loss(self):
        """Combined loss function with weighted components"""
        recon_loss = self._compute_reconstruction_loss()
        reg_loss = self._compute_regularization_loss()
        struct_loss = self._compute_structural_loss()

        total_loss = recon_loss + \
                    self.coeff_reg * reg_loss + \
                    self.lam * struct_loss

        self.loss_components = {
            'reconstruction': recon_loss.item(),
            'regularization': reg_loss.item(),
            'structural': struct_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss

    def update_graph(self, embedding):
        embedding = embedding.detach()
        if embedding.device.type == 'cuda':
            try:
                return embedding.matmul(embedding.t())
            except RuntimeError as exc:
                msg = str(exc).lower()
                if 'out of memory' not in msg and 'cuda' not in msg:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('update_graph GPU path failed; falling back to CPU.')
        embedding_cpu = embedding.cpu()
        return embedding_cpu.matmul(embedding_cpu.t())

    def clustering(self, weights):
        try:
            if torch.isnan(weights).any() or torch.isinf(weights).any():
                weights = torch.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=-1e6)

            degree = torch.sum(weights, dim=1).pow(-0.5)
            L = (weights * degree).t() * degree

            try:
                _, vectors = torch.linalg.eigh(L)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if weights.device.type != 'cuda' or ('out of memory' not in msg and 'cuda' not in msg):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('clustering GPU eigh failed; falling back to CPU.')
                _, vectors = torch.linalg.eigh(L.cpu())

            indicator = vectors[:, -self.n_clusters:].detach()
            km = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=20,
                max_iter=1000,
            ).fit(indicator.cpu().numpy())

            labels = km.labels_
            acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), labels)
            return acc, nmi, ari, f1

        except Exception as e:
            print(f"Clustering failed with error: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0

    def run(self):
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []
        eval_every = max(1, int(self.eval_every))
        self.best_embedding = self.embedding.detach().clone()
        self.train_time_sec = 0.0
        self.eval_time_sec = 0.0

        if self.skip_initial_eval or self.final_eval_only:
            best_acc, best_nmi, best_ari, best_f1 = 0.0, 0.0, 0.0, 0.0
            if self.final_eval_only:
                print('Initial evaluation skipped (final_eval_only enabled).')
            else:
                print('Initial evaluation skipped.')
        else:
            eval_start_time = time.perf_counter()
            with torch.no_grad():
                embedding = self.embedding.detach()
                weights = self.update_graph(embedding)
                laplacian_weights = get_Laplacian_from_weights(weights)
                acc, nmi, ari, f1 = self.clustering(laplacian_weights)
            self.eval_time_sec += time.perf_counter() - eval_start_time
            best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1
            print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
            del weights, laplacian_weights

        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        laplacian_adj = get_Laplacian(self.adjacency)

        for epoch in range(self.max_epoch):
            last_loss_value = None
            train_epoch_start_time = time.perf_counter()
            for i in range(self.max_iter):
                optimizer.zero_grad(set_to_none=True)
                self(laplacian_adj)
                loss = self._build_loss()
                loss.backward()
                optimizer.step()
                last_loss_value = loss.item()
                objs.append(last_loss_value)
                self.embedding = self.embedding.detach()
                del loss
            self.train_time_sec += time.perf_counter() - train_epoch_start_time

            should_eval = epoch == self.max_epoch - 1
            if not self.final_eval_only:
                should_eval = should_eval or (epoch % eval_every == 0)

            if should_eval:
                eval_start_time = time.perf_counter()
                with torch.no_grad():
                    embedding = self.embedding.detach()
                    weights = self.update_graph(embedding)
                    laplacian_weights = get_Laplacian_from_weights(weights)
                    acc, nmi, ari, f1 = self.clustering(laplacian_weights)
                self.eval_time_sec += time.perf_counter() - eval_start_time
                print(f'{epoch}loss: {last_loss_value:.4f}, ACC: {acc * 100:.2f}, NMI: {nmi * 100:.2f}, ARI: {ari * 100:.2f}, F1: {f1 * 100:.2f}')

                if acc >= best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                    self.best_embedding = embedding.clone()
                acc_list.append(acc)
                nmi_list.append(nmi)
                ari_list.append(ari)
                f1_list.append(f1)
                del weights, laplacian_weights
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        timing_summary = {
            "pretrain_time_sec": self.pretrain_time_sec,
            "train_time_sec": self.train_time_sec,
            "eval_time_sec": self.eval_time_sec,
            "train_and_eval_time_sec": self.train_time_sec + self.eval_time_sec,
            "total_time_sec": self.pretrain_time_sec + self.train_time_sec + self.eval_time_sec,
            "final_eval_only": self.final_eval_only,
            "skip_initial_eval": self.skip_initial_eval,
            "max_epoch": self.max_epoch,
            "max_iter": self.max_iter,
        }
        print("TIMING_SUMMARY " + json.dumps(timing_summary, ensure_ascii=False, sort_keys=True))

        print("best_acc{},best_nmi{},best_ari{},best_f1{}".format(best_acc, best_nmi, best_ari, best_f1))
        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)
        print(acc_list.mean(), "±", acc_list.std())
        print(nmi_list.mean(), "±", nmi_list.std())
        print(ari_list.mean(), "±", ari_list.std())
        print(f1_list.mean(), "±", f1_list.std())
        return best_acc, best_nmi, best_ari, best_f1

    def build_pretrain_loss(self):
        loss = self._compute_reconstruction_loss(zero_diagonal=True)
        loss_reg = self._compute_regularization_loss()
        return loss + self.coeff_reg * loss_reg

    def pretrain(self, pretrain_iter, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        pretrain_start_time = time.perf_counter()
        
        Laplacian = get_Laplacian(self.adjacency).to(self.device)
        last_loss_value = None
        
        for i in range(pretrain_iter):
            optimizer.zero_grad(set_to_none=True)
            self(Laplacian)
            loss = self.build_pretrain_loss()
            loss.backward()
            optimizer.step()
            last_loss_value = loss.item()
            self.embedding = self.embedding.detach()
            del loss

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.pretrain_time_sec = time.perf_counter() - pretrain_start_time
        print(last_loss_value)


class AGREE(GCGQ):
    """Public-facing model name for the cleaned release."""

    pass

