
from typing import List, Optional, Union
import numpy as np
from tqdm import tqdm
import scanpy as sc
import faiss
from sklearn.decomposition import PCA
import networkx as nx
from community import community_louvain
import math
import bisect
import colorsys

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)
#         sc.pp.filter_cells(adata, min_genes=200)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if normalize_input:
        sc.pp.scale(adata)
    return adata
    
def fast_decompose_clusters(data, k=15, seed=2024):
    if k+1 > data.shape[0]:
        k = 5
    if min(data.shape[0], data.shape[1]) >= 50:
        nps = 50
    else:
        nps = min(data.shape[0], data.shape[1])
    pca = PCA(n_components=nps, random_state=seed)
    sub_data = pca.fit_transform(data)
    index = faiss.IndexFlatL2(sub_data.shape[1]) 
    index.add(sub_data)
    D, I = index.search(sub_data, k+1)
    G = nx.Graph()
    for i in range(I.shape[0]):
        for j in I[i][1:]:
            G.add_edge(i, j)
    partition = community_louvain.best_partition(G, random_state=seed)
    return partition


def Finding_DEGs(data, indices1, indices2, neighbor_k=2, seed=2024, n_top=None):
    tmp_X = data[indices1, :]
    expr_gs = (tmp_X>0).sum(axis=0)
    h = np.median(expr_gs)
    re_gs = np.where(expr_gs >= h)[0]
    tmp_X = data[:, re_gs].copy()
    mnm1 = np.mean(tmp_X[indices1, :], axis=0)
    mnm2 = np.mean(tmp_X[indices2, :], axis=0)
    diff = np.array([np.log2(a + 1) - np.log2(b + 1) for a, b in zip(mnm1, mnm2)])
    nonzeros = len(np.where(diff>0)[0])
    if n_top is None:
        n_top = 100
    n_top = min(nonzeros, n_top)
    degs = np.array(re_gs)[np.argsort(-diff)][:n_top]
    return degs


def sum_series(p, d):
   # tail recursive helper function
   def helper(ret, p, d, i):
       term = math.pow(p, i)/i
       if d == i:
           return ret + term
       return helper(ret + term, p, d, i+1)
   return helper(0, p, d, 1)


def generate_hex_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 1.0
        value = 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        r_int = int(r * 255)
        g_int = int(g * 255)
        b_int = int(b * 255)
        hex_color = "#{:02x}{:02x}{:02x}".format(r_int, g_int, b_int)
        colors.append(hex_color)
    return colors
    
    
class RankingSimilarity:
    """
    This class will include some similarity measures between two different
    ranked lists.
    """
    def __init__(
        self,
        S: Union[List, np.ndarray],
        T: Union[List, np.ndarray],
        verbose: bool = False,
    ) -> None:
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ["a", "b", "c", "d", "e"]
        T = ["b", "a", 1, "d"]

        Both lists reflect the ranking of the items of interest, for example,
        list S tells us that item "a" is ranked first, "b" is ranked second,
        etc.

        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element"s position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
            verbose: If True, print out intermediate results. Default to False.
        """

        assert type(S) in [list, np.ndarray]
        assert type(T) in [list, np.ndarray]

        assert len(S) == len(set(S))
        assert len(T) == len(set(T))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder

    def assert_p(self, p: float) -> None:
        """Make sure p is between (0, 1), if so, assign it to self.p.

        Args:
            p (float): The value p.
        """
        assert 0.0 < p < 1.0, "p must be between (0, 1)"
        self.p = p

    def _bound_range(self, value: float) -> float:
        """Bounds the value to [0.0, 1.0]."""

        try:
            assert (0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print("Value out of [0, 1] bound, will bound it.")
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one

    def rbo(
        self,
        k: Optional[float] = None,
        p: float = 1.0,
        ext: bool = False,
    ) -> float:
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the
        RBO paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        If p = 1, it returns to the un-bounded set-intersection overlap,
        according to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it
        essentially control the "top-weightness". Simply put, to an extreme,
        a small p value will only consider first few items, whereas a larger p
        value will consider more items. See Eq. (21) for quantitative measure.

        Args:
            k: The depth of evaluation.
            p: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext: If True, we will extrapolate the rbo, as in Eq. (23).

        Returns:
            The rbo at depth k (or extrapolated beyond).
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float("inf")
        k = min(self.N_S, self.N_T, k)

        # initialize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            self.assert_p(p)
            weights = [1.0 * (1 - p) * p**d for d in range(k)]

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        for d in tqdm(range(1, k), disable=~self.verbose):

            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases did not happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p**k)

        return self._bound_range(AO[-1])

    def top_weightness(
            self,
            p: Optional[float] = None,
            d: Optional[int] = None):
        """
        This function will evaluate the degree of the top-weightness of the
        rbo. It is the implementation of Eq. (21) of the rbo paper.

        As a sanity check (per the rbo paper),
        top_weightness(p=0.9, d=10) should be 86%
        top_weightness(p=0.98, d=50) should be 86% too

        Args:
            p (float), default None: A value between zero and one.
            d (int), default None: Evaluation depth of the list.

        Returns:
            A float between [0, 1], that indicates the top-weightness.
        """

        # sanity check
        self.assert_p(p)

        if d is None:
            d = min(self.N_S, self.N_T)
        else:
            d = min(self.N_S, self.N_T, int(d))

        if d == 0:
            top_w = 1
        elif d == 1:
            top_w = 1 - 1 + 1.0 * (1 - p) / p * (np.log(1.0 / (1 - p)))
        else:
            sum_1 = 0
            for i in range(1, d):
                sum_1 += 1.0 * p**(i) / i
            top_w = 1 - p**(i) + 1.0 * (1 - p) / p * (i + 1) * \
                (np.log(1.0 / (1 - p)) - sum_1)  # here i == d-1

        if self.verbose:
            print("The first {} ranks have {:6.3%} of the weight of "
                  "the evaluation.".format(d, top_w))

        return self._bound_range(top_w)
