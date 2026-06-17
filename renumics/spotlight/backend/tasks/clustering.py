from typing import List, Tuple
import numpy as np
from renumics.spotlight.data_store import DataStore
from renumics.spotlight.backend.tasks.reduction import align_data

def compute_hdbscan(
    data_store: DataStore,
    column_names: List[str],
    indices: List[int],
    min_cluster_size: int = 3,
    pca_dim: int =40,
    umap_dim: int =5
) -> Tuple[np.ndarray, List[int]]:
    import hdbscan

    data, indices = align_data(data_store, column_names, indices)

    if data.size == 0:
        return np.array([]), []
    
    from sklearn.decomposition import PCA
    from umap import UMAP

    pca_reduced_data = PCA(n_components=min(pca_dim, len(data[0]))).fit_transform(data)
    umap_reduced_data = UMAP(n_components=umap_dim, min_dist=0.0, n_neighbors=70, random_state=42).fit_transform(pca_reduced_data)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=2
    ).fit_predict(umap_reduced_data)

    return labels, indices

def compute_leiden(
    data_store: DataStore,
    column_names: List[str],
    indices: List[int],
    k: int = 15,
    resolution: float = 1.0,
    pca_dim: int = 50,
) -> Tuple[np.ndarray, List[int]]:
    import igraph as ig
    import leidenalg
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    data, indices = align_data(data_store, column_names, indices)

    if data.size == 0:
        return np.array([]), []

    # Reduce dimensions before building the graph
    data = PCA(n_components=min(pca_dim, data.shape[1])).fit_transform(data)

    # Build k-NN graph in PCA space
    nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(data)
    distances, neighbor_indices = nn.kneighbors(data)

    sources = np.repeat(np.arange(len(data)), k)
    targets = neighbor_indices.flatten()
    weights = (1 - distances.flatten()).tolist()  # cosine similarity as edge weight

    g = ig.Graph(
        n=len(data),
        edges=list(zip(sources.tolist(), targets.tolist())),
        directed=False,
    )
    g.es['weight'] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,  # supports resolution param
        resolution_parameter=resolution,
        weights='weight',
        seed=42,
    )

    labels = np.array(partition.membership)  # 0-indexed, no outlier label
    return labels, indices


def compute_evoc(
    data_store: DataStore,
    column_names: List[str],
    indices: List[int],
) -> Tuple[np.ndarray, List[int]]:
    import evoc

    clusterer = evoc.EVoC(random_state=42)

    data, indices = align_data(data_store, column_names, indices)

    if data.size == 0:
        return np.array([]), []


    cluster_labels = clusterer.fit_predict(data)

    return cluster_labels, indices
