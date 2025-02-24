
import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx
from community import community_louvain
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
import time
from utils_functions import *
import os
import json
import pydot
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler
import anndata
import matplotlib.patches as mpatches

class MusTree:
	"""
	scMustree: Inference and visualization of multi-scale cell tree for decoding functional diversity.
	"""
	def __init__(self, data_matrix: np.ndarray, dataName: str = None, cellNames: list = None, geneNames: list = None, seed: int = None, save_path: str = None):
		"""
		Initialize the object with expression data.
		Parameters
		----------
		data_matrix : `2-D numpy.array` or 'anndata object'
		The expression data matrix with genes in columns and cells in rows. or Annotated data matrix.
		dataName : string
		Name of scRNA-seq dataset. (default: None)
		cellNames : list -> string
		The length must be the same as the number of rows in the data matrix.
		Names of all cells. (default: None)
		geneNames : list -> string
		The length must be the same as the number of columns in the data matrix.
		Names of all genes. (default: None)
		normalization : boolean
		Whether the data needs to be normalized. (default: True)
		seed : integer
		Random seed. (default: 2024)
		save_path : string
		Path to save results.
		"""
		if isinstance(data_matrix, np.ndarray):
			if data_matrix.ndim != 2:
				raise ValueError("data_matrix is not a 2D np.ndarray.")
		elif isinstance(data_matrix, anndata.AnnData):
			if not isinstance(data_matrix.X, np.ndarray):
				data_matrix.X = data_matrix.X.toarray()
			if data_matrix.shape[0] == 0 or data_matrix.shape[1] == 0:
				raise ValueError("data_matrix in AnnData is empty or has invalid shape.")
		else:
			raise TypeError("data_matrix should be either a 2D np.ndarray or an anndata.AnnData object.")
		if dataName is not None:
			assert isinstance(dataName, str), "dataName must be a string"
		
		n_cells, n_genes = data_matrix.shape
		if isinstance(data_matrix, np.ndarray):
			if cellNames is not None:
				assert isinstance(cellNames, list), "cellNames must be a list"
				assert len(cellNames) == n_cells, f"Number of cell names ({len(cellNames)}) must match the number of rows ({n_cells}) in data_matrix"
			if geneNames is not None:
				assert isinstance(geneNames, list), "geneNames must be a list"
				assert len(geneNames) == n_genes, f"Number of gene names ({len(geneNames)}) must match the number of columns ({n_genes}) in data_matrix"
			self.data_matrix = data_matrix
			self.cellNames = cellNames if cellNames is not None else [f"cell{i+1}" for i in range(n_cells)]
			self.geneNames = geneNames if geneNames is not None else [f"gene{i+1}" for i in range(n_genes)]
			self.adata = sc.AnnData(self.data_matrix)
			self.adata.var_names = self.geneNames
			self.adata.obs_names = self.cellNames
			
		else:
			self.data_matrix = data_matrix.X
			self.cellNames = data_matrix.obs_names
			self.geneNames = data_matrix.var_names
			self.adata = data_matrix
		
		if seed is not None:
			assert isinstance(seed, int), "seed must be an integer"
		if save_path is not None:
			assert isinstance(save_path, str), "save_path must be a string"
			assert os.path.isdir(save_path), f"save_path '{save_path}' is not a valid directory"

		self.dataName = dataName if dataName is not None else "analysis_data"
		self.seed = seed if seed is not None else 2024
		self.save_path = save_path if save_path is not None else './'
		self.n_cells = n_cells
		self.n_genes = n_genes
		self.sub_clusters = None
		self.clusters_degs = {}
		self.cluster_specific_cell_score = {}
		self.cluster_specific_scores = {}
		self.G = None
		self.nodes_to_cellids = {}
		self.positions = None
		self.umap_embedding = None

	def preprocess(self):
		print(">>> Data preprocessing in progress...")
		self.adata = normalize(self.adata)
		self.data_matrix = self.adata.X.copy()
		self.n_cells = self.data_matrix.shape[0]
		self.n_genes = self.data_matrix.shape[1]
		print(">>> After preprocessing, cells: %d; genes: %d;" % (self.n_cells, self.n_genes))


	def cluster_decomposition(self):
		print(">>> Performing cluster decomposition...")
		initial_clusters = fast_decompose_clusters(data=self.data_matrix, seed=self.seed)
		initial_clusters = np.array([initial_clusters[node] for node in range(self.n_cells)])
		pseudo_subclusters = initial_clusters.copy()
		h1 = max(20, int(0.01*self.n_cells))
		cluster_dict = Counter(pseudo_subclusters)
		depths = {}
		dpt = 2
		iter_max = 0
		while iter_max < 3:
			depths.update((key, dpt) for key in list(set([i for i, count in cluster_dict.items() if count < h1]) - set(depths.keys())))
			dpt = dpt + 1
			c_max = max(pseudo_subclusters) + 1
			c_list = list(set([i for i, count in cluster_dict.items() if count >= h1]) - set(depths.keys()))
			if len(c_list) == 0:
				break

			for clustid in c_list:
				idx = np.where(pseudo_subclusters == clustid)[0]
				idx_ = np.where(pseudo_subclusters != clustid)[0]
				temp_X = self.data_matrix[idx, :].copy()
				temp_clusters = fast_decompose_clusters(data=temp_X, seed=self.seed)
				temp_clusters = [temp_clusters[node] + c_max for node in range(temp_X.shape[0])]
			
				if len(np.unique(temp_clusters)) != 1:
					pseudo_subclusters[idx] = temp_clusters
					c_max = max(pseudo_subclusters) + 1
				else:
					depths[clustid] = dpt - 1
			iter_max = iter_max + 1
			cluster_dict = Counter(pseudo_subclusters)
		
		_, pseudo_subclusters = np.unique(pseudo_subclusters, return_inverse=True)
		self.sub_clusters = pseudo_subclusters.copy()
		n_clusters = len(set(pseudo_subclusters))
		print(f">>> Decomposition resulted in {n_clusters} clusters.")
		
		
	def infer_multiscale_tree(self):
		print(">>> Detecting cluster-specific DEGs and scores...")
		if self.sub_clusters is None:
			raise ValueError("Please run the function cluster_decomposition() first!")
		c_list = list(set(self.sub_clusters))
		n_clusters = len(c_list)
		IFmodel = IsolationForest(n_estimators=100, random_state=self.seed, n_jobs=-1)
		for i in tqdm(range(n_clusters)):
			idx = np.where(self.sub_clusters == c_list[i])[0]
			idx_ = np.where(self.sub_clusters != c_list[i])[0]
			degs = Finding_DEGs(data=self.data_matrix, indices1=idx, indices2=idx_, n_top=100)
			self.clusters_degs[c_list[i]] = degs.copy()
			
			IFmodel.fit(self.data_matrix[:, degs])
			s = IFmodel.score_samples(self.data_matrix[:, degs])
			self.cluster_specific_cell_score[c_list[i]] = s.copy()
			cluster_specific_score = []
			for x in list(set(self.sub_clusters)):
				tmp_idx = np.where(self.sub_clusters==x)[0]
				cluster_specific_score.append(np.median(s[tmp_idx]))
			
			self.cluster_specific_scores[c_list[i]] = cluster_specific_score.copy()
		
		print(">>> Calculating initial inter-cluster distances...")
		d = 5
		for p in [round(i, 2) for i in np.arange(0.99, 0.5, -0.01)]:
			wrbo1_d = 1 - math.pow(p, d-1) + (((1-p)/p) * d *(np.log(1/(1-p)) - sum_series(p, d-1)))
			if wrbo1_d>=0.9:
				break
		
		distance_matrix = np.zeros((len(c_list), len(c_list)))
		for i in tqdm(range(n_clusters-1)):
			s = self.cluster_specific_scores[c_list[i]]
			for j in range(i+1, n_clusters):
				s_ = self.cluster_specific_scores[c_list[j]]
				sim = RankingSimilarity(np.argsort(s), np.argsort(s_)).rbo(p=p)
				distance_matrix[i, j] = 1-sim
				distance_matrix[j, i] = 1-sim
		
		print(">>> Starting bottom-up tree inference...")
		
		comb_dist_mat = distance_matrix.copy()
		sub_clusters = self.sub_clusters.copy()
		self.G = nx.DiGraph()
		self.G.add_nodes_from(list(set(sub_clusters)))
		for i in set(sub_clusters):
			self.nodes_to_cellids[i] = list(np.where(sub_clusters==i)[0])
		
		cnames = list(set(sub_clusters))
		hdists = [round(i, 2) for i in np.arange(0.01, 2, 0.01)]
		with tqdm(total=len(set(self.sub_clusters)), desc="Processing nodes") as pbar:
			while comb_dist_mat.shape[0] > 1:
				np.fill_diagonal(comb_dist_mat, np.inf)
				for hd in hdists:
					positions = np.argwhere(comb_dist_mat <= hd)
					if positions.size > 0:
						hd_used = hd
						break
				comb_cls = []
				for pair in positions:
					comb_cls.append(cnames[pair[0]])
					comb_cls.append(cnames[pair[1]])
					
				comb_cls = list(set(comb_cls))
				adjacency_matrix = np.zeros((len(comb_cls), len(comb_cls)), dtype=int)
				for pair in positions:
					adjacency_matrix[comb_cls.index(cnames[pair[0]]), comb_cls.index(cnames[pair[1]])] = 1
					adjacency_matrix[comb_cls.index(cnames[pair[1]]), comb_cls.index(cnames[pair[0]])] = 1
				G_ = nx.from_numpy_array(adjacency_matrix)
				connected_components = list(nx.connected_components(G_))
				comb_list = []
				for pair in connected_components:
					comb_list.append([comb_cls[i] for i in pair])
				c_max = max(sub_clusters)+1
				
				
				new_list = []
				comb_list = [lst for lst in comb_list if len(lst) > 1]
				for comb in comb_list:
					self.nodes_to_cellids[c_max] = []
					tmp_idx = [cnames.index(i) for i in comb]
					sub_matrix = comb_dist_mat[np.ix_(tmp_idx, tmp_idx)]
					positions = np.where(sub_matrix > hd_used+0.02)
					if len(positions[0])>0:
						outliers = list(set(list(positions[0]) + list(positions[1])))
						remainers = list(set(list(range(len(tmp_idx)))) - set(outliers))
						if len(remainers) == 0:
							min_index = np.unravel_index(np.argmin(sub_matrix, axis=None), sub_matrix.shape)
							comb = [np.array(cnames)[np.array(tmp_idx)[min_index[0]]], np.array(cnames)[np.array(tmp_idx)[min_index[1]]]]
						elif len(remainers) == 1:
							voting = np.argmin(sub_matrix[remainers, :])
							comb = [np.array(cnames)[np.array(tmp_idx)[remainers]],np.array(cnames)[np.array(tmp_idx)[voting]]]
						else:
							comb = [np.array(cnames)[np.array(tmp_idx)[remainers]]]
					tmp_idx = [cnames.index(i) for i in comb]
					sub_matrix = comb_dist_mat[np.ix_(tmp_idx, tmp_idx)]
					upper_triangle_indices = np.triu_indices(len(sub_matrix), k=1)
					upper_triangle_elements = sub_matrix[upper_triangle_indices]
					
					for c in comb:
						self.G.add_edge(c, c_max, dist=np.median(upper_triangle_elements))
						idx = np.where(sub_clusters == c)[0]
						self.nodes_to_cellids[c_max] = self.nodes_to_cellids[c_max] + self.nodes_to_cellids[c]
						sub_clusters[idx] = c_max
						
					new_list.append(c_max)
					c_max = max(sub_clusters)+1
					
				if len(set(sub_clusters))<2:
					break
				
				prev_cnames = cnames.copy()
				cnames = list(set(sub_clusters))
				n_clusters = len(set(sub_clusters))
				
				for i in range(len(new_list)):
					idx = np.where(sub_clusters == new_list[i])[0]
					idx_ = np.where(sub_clusters != new_list[i])[0]
					degs = Finding_DEGs(data=self.data_matrix, indices1=idx, indices2=idx_, n_top=100)
					self.clusters_degs[new_list[i]] = degs.copy()
					IFmodel.fit(self.data_matrix[:, degs])
					s = IFmodel.score_samples(self.data_matrix[:, degs])
					self.cluster_specific_cell_score[new_list[i]] = s.copy()
					
				tmp_dist_matrix = np.zeros((n_clusters,n_clusters))
				for i in range(n_clusters):
					cluster_specific_score = []
					s = self.cluster_specific_cell_score[cnames[i]]
					for x in cnames:
						tmp_idx = np.where(sub_clusters==x)[0]
						cluster_specific_score.append(np.median(s[tmp_idx]))

					self.cluster_specific_scores[cnames[i]] = cluster_specific_score.copy()
					
				for i in range(n_clusters-1):
					s = self.cluster_specific_scores[cnames[i]]
					for j in range(i+1, n_clusters):
						s_ = self.cluster_specific_scores[cnames[j]]
						sim = RankingSimilarity(np.argsort(s), np.argsort(s_)).rbo(p=p)
						tmp_dist_matrix[i, j] = 1-sim
						tmp_dist_matrix[j, i] = 1-sim
					
				comb_dist_mat = tmp_dist_matrix.copy()
				# pbar.update(len(set(self.sub_clusters))-comb_dist_mat.shape[0])
				pbar.n = len(set(self.sub_clusters))-comb_dist_mat.shape[0]+2
				pbar.last_print_n = len(set(self.sub_clusters))-comb_dist_mat.shape[0]+2
				pbar.update(0)
				pbar.set_postfix_str(f"Current: {comb_dist_mat.shape[0]-2}")
		
		self.G = self.G.reverse()
		self.positions = nx.nx_agraph.graphviz_layout(self.G, prog="dot")
		
		print(f">>> Results are being output and saved to {self.save_path}.")
		dataName = self.dataName
		nx.write_gexf(self.G, self.save_path+dataName+'_dendrogram.gexf')
		
		file_path = self.save_path+dataName+'_gnames.txt'
		with open(file_path, "w", encoding="utf-8") as file:
			for item in list(self.adata.var_names):
				file.write(item + "\n")

		nodes_to_cellids = {int(k):v for k,v in self.nodes_to_cellids.items()}
		for k,v in nodes_to_cellids.items():
			nodes_to_cellids[k] = [int(i) for i in v]
		json.dump(nodes_to_cellids, open(self.save_path+dataName+'_dendrogram_nodes_id.txt','w'))

		clusters_degs = {int(k):v for k,v in self.clusters_degs.items()}
		for k,v in clusters_degs.items():
			clusters_degs[k] = [int(i) for i in v]
		json.dump(clusters_degs, open(self.save_path+dataName+'_clustersDEGs.txt','w'))


	def read_tree(self, path=None):
		path = path if path is not None else self.save_path
		if not os.path.exists(path):
			raise FileNotFoundError(f"The path {path} does not exist.")
		
		file_path1 = os.path.join(path, self.dataName+'_dendrogram.gexf')
		file_path2 = os.path.join(path, self.dataName+'_dendrogram_nodes_id.txt')
		if not os.path.isfile(file_path1):
			raise FileNotFoundError(f"The file {file_path1} does not exist in the path {path}.")
		if not os.path.isfile(file_path2):
			raise FileNotFoundError(f"The file {file_path2} does not exist in the path {path}.")
		
		self.G = nx.read_gexf(file_path1)
		self.G = nx.relabel_nodes(self.G, {node: int(node) for node in self.G.nodes()})
		file = open(file_path2)
		nodes_to_cellids = json.loads(file.readline())
		self.nodes_to_cellids = {int(k):v for k,v in nodes_to_cellids.items()}
		self.positions = nx.nx_agraph.graphviz_layout(self.G, prog="dot")
		
		
	def plot_tree(self, path=None, label=None, remain_types=None, 
				width=None, height=None, font_size=None, node_size=None, color_list=None):
		if self.G is None:
			raise TypeError("The object does not contain a tree structure.")
		
		path = path if path is not None else self.save_path
		if not os.path.exists(path):
			raise FileNotFoundError(f"The path {path} does not exist.")
		if label is not None:
			assert len(label) == self.n_cells, f"Number of cells in label ({len(label)}) must match the number of cells ({self.n_cells}) in data"
		if remain_types is not None and label is None:
			raise ValueError("The 'remain_types' parameter can only be used if 'label' is provided.")
		if remain_types is not None and not set(remain_types).issubset(set(label)):
			raise ValueError("All elements of 'remain_types' must be in the 'label' set.")
		for dim, value in [("width", width), ("height", height), ("font_size", font_size), ("node_size", node_size)]:
			if value is not None:
				if not isinstance(value, (int, float)):
					raise ValueError(f"The '{dim}' parameter must be an integer or float.")
				if value <= 0:
					raise ValueError(f"The '{dim}' parameter must be a positive number.")	
		if color_list is not None:
			hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
			for color in color_list:
				if not hex_pattern.match(color):
					raise ValueError(f"Invalid color code in color_list: {color}. Must be a valid 6-digit hex code starting with '#'.")
		if color_list is not None and label is None:
			print("Warning: color_list will only use the first color since label is None.")
		if color_list is not None and label is not None:
			if remain_types is not None:
				if len(color_list) < len(set(remain_types)):
					raise ValueError(f"color_list should have at least {len(set(remain_types))}.")
			else:
				if len(color_list) < len(set(label)):
					raise ValueError(f"color_list should have at least {len(set(label))}.")
		
		width = width if width is not None else 6  # Default width in cm
		height = height if height is not None else 6  # Default height in cm
		font_size = font_size if font_size is not None else 1  # Default width in cm
		node_size = node_size if node_size is not None else 3 # Default height in cm
		fig_width_inch = width / 2.54  
		fig_height_inch = height / 2.54 
		
		if label is not None:
			types = list(set(label))
			typetoid = dict(zip(types, range(1, len(types)+1)))
			color_list = color_list if color_list is not None else ['#e5e5e5']+generate_hex_colors(len(types))
			groups = []
			if remain_types is None:
				remain_types = types
				fn = "dendrogram_withlabels"
			else:
				fn = "dendrogram_with_"+'_'.join(remain_types)+"_labels"
			remain_types = list(set(remain_types))
			
			for i in list(self.G.nodes):
				idx = self.nodes_to_cellids[int(i)]
				count = Counter(label[idx])
				if count.most_common(1)[0][0] in remain_types:
					self.G.nodes[i]['group'] = typetoid[count.most_common(1)[0][0]]
					groups.append(typetoid[count.most_common(1)[0][0]])
				else:
					self.G.nodes[i]['group'] = 0
					groups.append(0)

			node_colors = []
			for g in groups:
				if g == 0:
					node_colors.append(color_list[0]) 
				else:
					node_colors.append(color_list[g % len(color_list)])
			
			plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=2000)
			ax = plt.gca()
			nx.draw(self.G, self.positions, with_labels=True, font_size=font_size, node_size=node_size, 
					node_color=node_colors, arrowsize=0.1, width=0.05, font_color='black')
					
			legend_labels = remain_types
			legend_colors = [color_list[typetoid[i]] for i in remain_types]
			handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
			legend_params = {'handles': handles, 'bbox_to_anchor': (1.1, 1),  'loc': 'upper left', 'ncol': 1,
							'borderaxespad': 0.3, 'borderpad': 0.3, 'labelspacing': 0.3, 'handlelength': 0.5, 
							'handletextpad': 0.3, 'prop': {'family': 'Arial', 'size': 6}, 'frameon': False}
			legend = ax.legend(**legend_params)
			plt.subplots_adjust(right=0.8)
					
			# plt.savefig(fname=path+self.dataName+'_'+fn+'.png')
			plt.savefig(fname=path+self.dataName+'_'+fn+'.png', bbox_inches='tight',  dpi = 2000, pad_inches=0.05)
			plt.show()
			
		else:
			color_list = color_list if color_list is not None else ['#e5e5e5']
			plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=2000)
			nx.draw(self.G, self.positions, with_labels=True, font_size=font_size, node_size=node_size, 
					node_color=color_list[0], arrowsize=0.1, width=0.05, font_color='black')
			plt.savefig(fname=path+self.dataName+'_dendrogram.png', bbox_inches='tight',  dpi = 2000, pad_inches=0.05)
			plt.show()
	
	
	def nodes_degs_analysis(self, path=None, nodeid=None, ref_node_id=None):
		path = path if path is not None else self.save_path
		if not os.path.exists(path):
			raise FileNotFoundError(f"The path {path} does not exist.")
			
		if nodeid is None:
			raise ValueError("nodeid cannot be None.")
		elif nodeid not in [int(i) for i in list(self.G.nodes)]:
			raise ValueError(f"nodeid {nodeid} is not present in the tree.")

		idx1 = self.nodes_to_cellids[nodeid]
		tmp_label = np.array(['Others' for _ in range(self.n_cells)])
		tmp_label[idx1] = 'T1'
		adata = self.adata
		
		if ref_node_id is None:
			fn = str(nodeid)
			adata.obs['group'] = tmp_label
			sc.tl.rank_genes_groups(adata, groups=['T1'], groupby="group", method="wilcoxon", pts=True)
		else:
			if ref_node_id not in list(self.G.nodes):
				raise ValueError(f"ref_node_id {ref_node_id} is not present in the tree.")
			else:
				fn = str(nodeid)+'_vs_'+str(ref_node_id)
				idx2 = self.nodes_to_cellids[ref_node_id]
				tmp_label[idx2] = 'T2'
				adata.obs['group'] = tmp_label
				sc.tl.rank_genes_groups(adata, groups=['T1'], groupby="group", method="wilcoxon", reference='T2', pts=True)
		
		degsdf = sc.get.rank_genes_groups_df(adata, group='T1', log2fc_min=1.5, pval_cutoff=0.05)
		degsdf.to_csv(path+self.dataName+'_nodeid_'+fn+'_degs.csv', index=False)
	
	
	def plot_gene_expression_on_tree(self, path=None, markers_list=None, markers_name=None,
				width=None, height=None, font_size=None, node_size=None):
		
		adata = self.adata[:,~self.adata.var_names.duplicated()]
		
		if self.G is None:
			raise TypeError("The object does not contain a tree structure.")
			
		path = path if path is not None else self.save_path
		if not os.path.exists(path):
			raise FileNotFoundError(f"The path {path} does not exist.")
		if markers_list is not None:
			if not all(isinstance(marker, str) for marker in markers_list):
				raise ValueError("All elements in markers_list must be strings.")
			markers_list = list(set(markers_list).intersection(set(adata.var_names)))
			if len(markers_list) == 0:
				raise ValueError("None of the genes in markers_list exist in the data.")
			else:
				print(f">>>Number of matching genes: {len(markers_list)}")
				print(f">>>Matching genes: {', '.join(markers_list)}")
		else:
			raise ValueError("markers_list must be provided and cannot be None.")
		
		markers_name = markers_name if markers_name is not None else markers_list[0]+'_AndMore'
		
		for dim, value in [("width", width), ("height", height), ("font_size", font_size), ("node_size", node_size)]:
			if value is not None:
				if not isinstance(value, (int, float)):
					raise ValueError(f"The '{dim}' parameter must be an integer or float.")
				if value <= 0:
					raise ValueError(f"The '{dim}' parameter must be a positive number.")
		
		groups = []
		X_sub = adata[:, markers_list].X.copy()
		gene_means = np.mean(X_sub, axis=0)
		gene_stds = np.std(X_sub, axis=0) 
		Z_scores = (X_sub - gene_means) / gene_stds
		expression_scores_list = []
		for i in list(self.G.nodes):
			idx = self.nodes_to_cellids[int(i)]
			Z_scores_selected_cells = Z_scores[idx, :]
			expression_scores = np.mean(Z_scores_selected_cells) 
			expression_scores_list.append(expression_scores)
		
		scaler = MinMaxScaler()
		normalized_values  = scaler.fit_transform(np.array(expression_scores_list).reshape(-1, 1)).flatten()
		num_bins = 10
		bins = np.linspace(0, 1, num_bins)
		group = np.digitize(normalized_values, bins) - 1
		colors = plt.cm.viridis(np.linspace(0, 1, num_bins))
		node_colors = [colors[g] for g in group]
		
		for i in range(len(list(self.G.nodes))):
			self.G.nodes[list(self.G.nodes)[i]]['group'] = group[i]
		
		width = width if width is not None else 6  # Default width in cm
		height = height if height is not None else 6  # Default height in cm
		font_size = font_size if font_size is not None else 1  # Default width in cm
		node_size = node_size if node_size is not None else 3 # Default height in cm
		fig_width_inch = width / 2.54  
		fig_height_inch = height / 2.54 
		
		plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=2000)
		nx.draw(self.G, self.positions, with_labels=True, font_size=font_size, node_size=node_size,
				node_color=node_colors, arrowsize=0.1, width=0.05, font_color='black')
		plt.savefig(fname=path+self.dataName+'_'+markers_name+'_expression_tree.png', dpi = 2000, bbox_inches = 'tight')
		plt.show()
		
	
	def map_tree_nodes_to_UMAP(self, path=None, nodes_list=None, label=None, umap_embedding=None, 
								color_list=None, width=None, height=None, node_size=None):
		path = path if path is not None else self.save_path
		if not os.path.exists(path):
			raise FileNotFoundError(f"The path {path} does not exist.")
			
		if nodes_list is not None:
			nodes_list = list(set(nodes_list))
			cell_idx = []
			for i in nodes_list:
				if i not in [int(i) for i in list(self.G.nodes)]:
					raise ValueError(f"nodeid {i} is not present in the tree.")
				else:
					cell_idx.extend(self.nodes_to_cellids[int(i)])
			if len(set(cell_idx)) != len(cell_idx):
				print(">>>There are duplicate cell IDs in the node list.")
				
			if color_list is not None:
				if len(color_list) < len(set(nodes_list)):
					raise ValueError(f"color_list should have at least {len(set(nodes_list))}.")
			
			if label is not None:
				print(">>>Warning: Both 'nodes_list' and 'label' were provided. Only 'nodes_list' will be used.")
		else:
			if label is None:
				raise ValueError("Either 'nodes_list' or 'label' must be provided and cannot be None.")
			else:
				if color_list is not None:
					if len(color_list) < len(set(label)):
						raise ValueError(f"color_list should have at least {len(set(label))}.")
				
		if color_list is not None:
			hex_pattern = re.compile(r'^#[0-9A-Fa-f]{6}$')
			for color in color_list:
				if not hex_pattern.match(color):
					raise ValueError(f"Invalid color code in color_list: {color}. Must be a valid 6-digit hex code starting with '#'.")
					
		if umap_embedding is None:
			if self.umap_embedding is None:
				sc.pp.highly_variable_genes(self.adata, n_top_genes=2000)
				adata = self.adata[:, self.adata.var.highly_variable]
				sc.pp.scale(adata)
				sc.tl.pca(adata)
				sc.pp.neighbors(adata)
				sc.tl.umap(adata)
				self.umap_embedding = adata.obsm['X_umap'].copy()
		else:
			if not isinstance(umap_embedding, np.ndarray):
				raise ValueError("The 'umap_embedding' must be a numpy array.")
			if umap_embedding.ndim != 2 or umap_embedding.shape[0] != self.n_cells or umap_embedding.shape[1] != 2:
				raise ValueError(f"The 'umap_embedding' must be a 2D numpy array with shape ({self.n_cells}, 2).")
			self.umap_embedding = umap_embedding
			
		for dim, value in [("width", width), ("height", height), ("node_size", node_size)]:
			if value is not None:
				if not isinstance(value, (int, float)):
					raise ValueError(f"The '{dim}' parameter must be an integer or float.")
				if value <= 0:
					raise ValueError(f"The '{dim}' parameter must be a positive number.")
		
		width = width if width is not None else 6  # Default width in cm
		height = height if height is not None else 6  # Default height in cm
		node_size = node_size if node_size is not None else 3 # Default height in cm
		fig_width_inch = width / 2.54  
		fig_height_inch = height / 2.54 
		
		if nodes_list is not None:
			cid = 1
			label = np.array([0 for _ in range(self.n_cells)])
			for i in nodes_list:
				idx = self.nodes_to_cellids[i]
				label[idx] = cid
				cid = cid+1
			
			color_list = color_list if color_list is not None else ['#e5e5e5']+generate_hex_colors(len(nodes_list))
			node_colors = [color_list[i] for i in label]
		else:
			typetoid = dict(zip(set(label), range(len(set(label)))))
			color_list = color_list if color_list is not None else generate_hex_colors(len(set(label)))
			node_colors = [color_list[typetoid[i]] for i in label]
		
		plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=2000)
		if nodes_list is not None:
			ax = plt.gca()
			plt.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], c=node_colors,  s=0.8, alpha=0.8, edgecolors='none')
			plt.axis('off')
			legend_labels = ['Node: '+str(i) for i in nodes_list]
			legend_colors = color_list[1:len(nodes_list)+1]
			handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
			legend_params = {'handles': handles, 'bbox_to_anchor': (1.1, 1),  'loc': 'upper left', 'ncol': 1,
							'borderaxespad': 0.3, 'borderpad': 0.3, 'labelspacing': 0.3, 'handlelength': 0.5, 
							'handletextpad': 0.3, 'prop': {'family': 'Arial', 'size': 6}, 'frameon': False}
			legend = ax.legend(**legend_params)
			plt.subplots_adjust(right=0.8)

			plt.savefig(fname=path+self.dataName+'_UMAP_with_nodesid_'+'_'.join([str(i) for i in nodes_list])+'.png', bbox_inches='tight', pad_inches=0.05)
			# plt.savefig(fname=path+self.dataName+'_UMAP_with_nodesid_'+'_'.join([str(i) for i in nodes_list])+'.png')
			plt.show()		
		else:
			ax = plt.gca()
			plt.scatter(self.umap_embedding[:, 0], self.umap_embedding[:, 1], c=node_colors,  s=0.8, alpha=0.8, edgecolors='none')
			plt.axis('off')
			legend_labels = list(set(label))
			legend_colors = [color_list[typetoid[i]] for i in list(set(label))]
			handles = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
			legend_params = {'handles': handles, 'bbox_to_anchor': (1.1, 1),  'loc': 'upper left', 'ncol': 1,
							'borderaxespad': 0.3, 'borderpad': 0.3, 'labelspacing': 0.3, 'handlelength': 0.5, 
							'handletextpad': 0.3, 'prop': {'family': 'Arial', 'size': 6}, 'frameon': False}
			legend = ax.legend(**legend_params)
			plt.subplots_adjust(right=0.8)

			plt.savefig(fname=path+self.dataName+'_UMAP_with_label.png', bbox_inches='tight', pad_inches=0.05)
			# plt.savefig(fname=path+self.dataName+'_UMAP_with_label.png')
			plt.show()
		




