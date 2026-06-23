import owlready2
from pykeen import triples
from pyparsing import common
from setuptools import dist
import torch
from tqdm import tqdm
from pykeen.triples import TriplesFactory
import warnings
from collections import Counter, defaultdict
import networkx as nx
import math
import torch
from typing import Literal
import pandas as pd
import numpy as np


class SchemaTranslator:
    def __init__(self, ontology_path, triples):
        
        ontology=owlready2.get_ontology(ontology_path).load()
        
        triples=triples
        
        class_instances=self.class_to_entity(ontology)
        
        relation_ranges=self.relation_to_scope(ontology, triples, extract="range")
        
        relation_domains=self.relation_to_scope(ontology, triples, extract="domain")
        
        self.csr_list_r, self.offsets_r=self.CSR_offset_precomputing(
            ontology,
            triples,
            class_instances,
            extract="range")
        
        self.csr_list_d, self.offsets_d=self.CSR_offset_precomputing(
            ontology,
            triples,
            class_instances,
            extract="domain")
        
        self.hr_scores, self.rt_scores=self.compute_false_negative_likelyhood(triples)
        
        self.similarity_matrix_r, self.empty_mask_r=self.build_similarity_matrix(ontology, relation_ranges)
        
        self.similarity_matrix_d, self.empty_mask_d=self.build_similarity_matrix(ontology, relation_domains)
        
        self.disjoint_with_matrix_r, self.empty_mask_disjoint_r=self.build_disjoint_with_matrix(ontology, relation_ranges)
        
        self.disjoint_with_matrix_d, self.empty_mask_disjoint_d=self.build_disjoint_with_matrix(ontology, relation_domains)
        return
    
    def class_to_entity(self, ontology):
        class_instances = {}
        for cls in tqdm(ontology.classes(), desc="Precomputing class instances"):
            instances = list(cls.instances())
            class_instances[cls] = instances
        return class_instances


    def unpack_unionof(self, domain_or_range):
        classes = []
        for r in domain_or_range:
            if hasattr(r, 'Classes'):  # it's an Or, unpack it
                classes.extend(r.Classes)
            else:
                classes.append(r)
        return classes

    def relation_to_scope(self, ontology, triples, extract: Literal["range", "domain"]):
        relation_scope = {}
        for prop in tqdm(ontology.object_properties(), desc="Precomputing relation scopes"):
            rel_id=triples.relation_to_id.get(prop.iri)
            if extract == "range":
                classes = [r.name for r in self.unpack_unionof(prop.range) if hasattr(r, 'name')]
            else:
                classes = [r.name for r in self.unpack_unionof(prop.domain) if hasattr(r, 'name')]
            if classes:
                relation_scope[rel_id] = classes
            else:
                relation_scope[rel_id] = []
        # we remove any 'None' keys
        relation_scope = {k: v for k, v in relation_scope.items() if k is not None}
        # handle cases where r is a simple range or a union of classes
        return relation_scope
        

    def CSR_offset_precomputing(self, ontology, triples, class_instances, extract: Literal["range", "domain"]):
        csr_list = []
        offsets=[0]
        current_offset = 0
        for prop_id in tqdm(range(triples.num_relations), desc=f"Precomputing CSR and offsets for {extract}"):
        # get property range
            prop=ontology.search_one(iri=triples.relation_id_to_label[prop_id])
            if extract == "range":
                scope_ = prop.range
            else:
                scope_ = prop.domain
        # search for the range in the class_instances dictionary
            scope_instances = []
            for r in scope_:
                r_instances = class_instances.get(r, [])
                scope_instances.extend(r_instances)
            scope_instances = list(set(scope_instances))  # remove duplicates
        #append the result in csr_list and the offset in offsets
            csr_list.extend(scope_instances)
            current_offset += len(scope_instances)
            offsets.append(current_offset)
        
        return csr_list, offsets
    

    def compute_false_negative_likelyhood(self, triples):
        # 1. Extract the columns as Python lists (fastest way to pull them out)
        h_array = triples.mapped_triples[:, 0].tolist()
        r_array = triples.mapped_triples[:, 1].tolist()
        t_array = triples.mapped_triples[:, 2].tolist()

        # Build a single dataframe with one row per triple (h, r, t)
        df = pd.DataFrame({'h': h_array, 'r': r_array, 't': t_array})

        # 2. Count absolute frequencies of (h, r) and (r, t) pairs
        hr_counts = df.groupby(['h', 'r']).size().rename('count').reset_index()
        rt_counts = df.groupby(['r', 't']).size().rename('count').reset_index()

        # 3. Compute MAX, MEDIAN and STD of the counts, grouped by relation r
        # ddof=0 -> population std 
        hr_stats = hr_counts.groupby('r')['count'].agg(
            max='max',
            median='median',
            std=lambda x: x.std(ddof=0)
            ).fillna(0)

        rt_stats = rt_counts.groupby('r')['count'].agg(
            max='max',
            median='median',
            std=lambda x: x.std(ddof=0)
        ).fillna(0)

        # 4. Merge the per-relation stats back onto each (h,r) / (r,t) row
        hr_counts = hr_counts.merge(hr_stats, on='r', suffixes=('', '_r'))
        rt_counts = rt_counts.merge(rt_stats, on='r', suffixes=('', '_r'))

        # 5. Compute the likelihood score (0 to 1), vectorized over the whole column
        # log(count(h,r)+1) / log(MAX_h'(h',r)+1)*lambda(r) where lambda(r) = 1/(1 + std/median)
        hr_counts['score'] = (np.log1p(hr_counts['count']) / np.log1p(hr_counts['max']))*(1/1+(hr_counts['std']/(hr_counts['median']+1e-6)))  # Adding a small epsilon to avoid division by zero
        # log(count(r,t)+1) / log(MAX_t'(r,t')+1)*lambda(r) where lambda(r) = 1/(1 + std/median)
        rt_counts['score'] = (np.log1p(rt_counts['count']) / np.log1p(rt_counts['max']))*(1/1+(rt_counts['std']/(rt_counts['median']+1e-6)))  # Adding a small epsilon to avoid division by zero

        # 6. Rebuild the final dictionaries: {(h, r): score} and {(r, t): score}
        hr_scores = dict(zip(zip(hr_counts['h'], hr_counts['r']), hr_counts['score']))
        rt_scores = dict(zip(zip(rt_counts['r'], rt_counts['t']), rt_counts['score']))

        return hr_scores, rt_scores
    
    def build_graph(self, ontology) -> nx.DiGraph:
        """Builds the DAG from the ontology's subClassOf hierarchy."""
        G = nx.DiGraph()
        root = owlready2.owl.Thing.name
 
        for cls in ontology.classes():
            for parent in cls.is_a:
            # Filter Restrictions and other OWL non-class constructs
                if isinstance(parent, type) and issubclass(parent, owlready2.owl.Thing):
                    G.add_edge(parent.name, cls.name)
 
    # Attach classes without explicit parent to owl:Thing
        for cls in ontology.classes():
            if cls.name not in G.nodes:
                G.add_edge(root, cls.name)
            elif G.in_degree(cls.name) == 0 and cls.name != root:
                G.add_edge(root, cls.name)
        return G
 
 
    def compute_ic(self, G: nx.DiGraph) -> dict:
        """
        Calcola l'IC strutturale di Seco per ogni nodo del DAG.
        IC(c) = 1 - log(|discendenti(c)| + 1) / log(N)
        """
        N = G.number_of_nodes()
        ic = {}
        for node in G.nodes:
            hypo = len(nx.descendants(G, node))
            ic[node] = 1 - (math.log(hypo + 1) / math.log(N))
        return ic
 
 
    def get_lca(self, G: nx.DiGraph, ic: dict, c1: str, c2: str):
        """
        Trova l'antenato comune con IC massimo (il più specifico).
        Restituisce None se non esiste un antenato comune.
        """
        anc1 = nx.ancestors(G, c1) | {c1}
        anc2 = nx.ancestors(G, c2) | {c2}
        common = anc1 & anc2
        if not common:
            return None
        return max(common, key=lambda n: ic.get(n, 0))
 
 
    def jiang_conrath_sim(self, ic: dict, G: nx.DiGraph, c1: str, c2: str) -> float:
        """
        Similarità Jiang & Conrath normalizzata in [0, 1].
 
        dist(c1, c2) = IC(c1) + IC(c2) - 2 * IC(LCA)
        sim(c1, c2)  = 1 - dist / 2
 
        Casi limite:
        - Stessa classe        → 0.0
        - Classe non nel grafo → 0.0
        - Nessun LCA comune    → 0.0
        """
        if c1 not in ic or c2 not in ic:
            return 0.0
        lca = self.get_lca(G, ic, c1, c2)
        if c1 == c2:
            return 1.0
        if lca is None:
            return 0.0
        dist = ic[c1] + ic[c2] - 2 * ic[lca]
        return 1.0 - (dist / 2.0)
 
 
    def build_similarity_matrix(self, ontology, relation_scope: dict) -> torch.Tensor:
        """
        Builds the r x r matrix of taxonomic similarity between the classes
        appearing in the ranges of relations.

        Args:
            ontology:         OWLready2 ontology object
            relation_ranges:  dict {relation_id: class_name} mapping
                              each relation to its range class(es)

        Returns:
            Tensor (r x r) with sim[i, j] = taxonomic similarity between
            the range of relation i and the range of relation j.
        """
        G = self.build_graph(ontology)
        ic = self.compute_ic(G)
    
        r = len(relation_scope)
        matrix = torch.full((r, r), 1e-20)
 
        for i, classes_i in tqdm(relation_scope.items(), desc="Building similarity matrix"):
            for j, classes_j in relation_scope.items():
                scores = [
                    self.jiang_conrath_sim(ic, G, ci, cj)
                    for ci in classes_i
                    for cj in classes_j
                ]
                matrix[i, j] = sum(scores) / (len(scores) + 1e-6)
                
        empty_mask = matrix.sum(dim=1) == 0  # If the row is all 0, the relation has no taxonomic similarity with any other relation
        matrix.fill_diagonal_(0)  # Set diagonal to 0 to avoid considering self-similarity
        matrix[empty_mask] = 1/r  # Set the row to 1/r to simulate random sampling when there is no taxonomic similarity
        
        return matrix, empty_mask
    
    def build_disjoint_with_matrix(self, ontology, relation_scope: dict) -> torch.Tensor:
        """Costruisce la matrice r x r di disjointness tra le classi,
        restituendo 1 se le classi di range di due relazioni sono disjoint e 0 altrimenti.
        se il range è multiplo, restituiamo 0, in quanto non possiamo affermare che tutte le classi siano disjoint con quelle dell'altra relazione.
        """
        
        r=len(relation_scope)
        matrix = torch.full((r, r), 0)# i create a matrix of number sufficiently small

        #we precompute the dijointness, so that there are less calls to the ontology
        
        lookup = {}
        for i, classes in tqdm(relation_scope.items(), desc="Precomputing disjoint_relations"):
            if (len(classes) != 1 or 
                    classes[0] in [None, "None", "Thing", "owl:Thing"]): #there are multiscope properties, not yet supported, we leave this challenge to future brave researchers
                lookup[i] = (None, set())
            else:
                cls = ontology.search_one(iri=f'*{classes[0]}')
                disjoints=list(cls.disjoints())[0].entities if len(list(cls.disjoints()))>0 else []
                lookup[i] = (cls, set(disjoints))
                
        for i, (cls_i, disjoint_i) in lookup.items():
            for j, (cls_j, _) in lookup.items():
                if cls_i is None or cls_j is None:
                    continue
                else:
                    matrix[i, j] = 1.0 if cls_j in disjoint_i else 0

        empty_mask = matrix.sum(dim=1) == 0  # if the row is all 0, it means that the relation has no disjointness with any other relation
        matrix[empty_mask] = 1/r  # we set the row to 1/r, to simulate random sampling, as there is no disjointness with any other relation
        return matrix, empty_mask

        '''
        TODO:
        - MODEL THE NON-RANGE CASE ALSO IN THE OFFSETS AND CSR, SO AS TO HAVE A UNIFORM REPRESENTATION FOR ALL
        RELATIONS
        '''