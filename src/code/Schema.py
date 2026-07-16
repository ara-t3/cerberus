import pickle
import owlready2
from pykeen import triples
from pyparsing import common
from rdflib import RDF, Graph
from rdflib import RDF
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
    def __init__(self, t_box, a_box, triples):
        
        t_box=owlready2.get_ontology(t_box).load()
        
        triples=triples
        
        class_instances=self.class_to_entity(a_box)
        
        relation_ranges=self.relation_to_scope(t_box, triples, extract="range")
        
        relation_domains=self.relation_to_scope(t_box, triples, extract="domain")
        
        self.csr_list_r, self.offsets_r=self.CSR_offset_precomputing(
            t_box,
            triples,
            class_instances,
            extract="range")
        
        self.csr_list_d, self.offsets_d=self.CSR_offset_precomputing(
            t_box,
            triples,
            class_instances,
            extract="domain")
        
        self.hr_scores, self.rt_scores=self.compute_false_negative_likelyhood(triples)
        
        self.similarity_matrix_r, self.empty_mask_r=self.build_similarity_matrix(t_box, relation_ranges)
        
        self.similarity_matrix_d, self.empty_mask_d=self.build_similarity_matrix(t_box, relation_domains)
        
        self.disjoint_with_matrix_r, self.empty_mask_disjoint_r=self.build_disjoint_with_matrix(t_box, relation_ranges)
        
        self.disjoint_with_matrix_d, self.empty_mask_disjoint_d=self.build_disjoint_with_matrix(t_box, relation_domains)
        return
    
    def class_to_entity(self, abox):
        g = Graph()
        print('parsing abox')
        g.parse(abox, format="xml")
        
        

        class_instances = defaultdict(list)

        for subj, _, cls in tqdm(g.triples((None, RDF.type, None)), desc="Precomputing class instances"):
            class_instances[str(cls)].append(str(subj))

        return class_instances


    def unpack_unionof(self, domain_or_range):
        classes = []
        for r in domain_or_range:
            if hasattr(r, 'Classes'):  # it's an Or, unpack it
                classes.extend(r.Classes)
            else:
                classes.append(r)
        return classes

    def relation_to_scope(self, t_box, triples, extract: Literal["range", "domain"]):
        relation_scope = {}
        for rel_id in tqdm(range(triples.num_relations), desc="Precomputing relation scopes"):
            rel_label = triples.relation_id_to_label[rel_id]
            prop = t_box.search_one(iri=f'*{rel_label}')
            if prop:
                if extract == "range":
                    classes = [r.name for r in self.unpack_unionof(prop.range) if hasattr(r, 'name')]
                else:
                    classes = [r.name for r in self.unpack_unionof(prop.domain) if hasattr(r, 'name')]
                
                relation_scope[rel_id] = classes
            else:
                relation_scope[rel_id] = []
        return relation_scope
        

    def CSR_offset_precomputing(self, t_box, triples, class_instances, extract: Literal["range", "domain"]):
        csr_list = []
        offsets=[0]
        current_offset = 0
        for prop_id in tqdm(range(triples.num_relations), desc=f"Precomputing CSR and offsets for {extract}"):
        # get property scope (range or domain) from the ontology
            prop=t_box.search_one(iri=f'*{triples.relation_id_to_label[prop_id]}')
            if extract == "range":
                scope_ = prop.range
            else:
                scope_ = prop.domain
        # search for the scope in the class_instances dictionary
            scope_instances = []
            scope_ = self.unpack_unionof(scope_)
            for r in scope_:
                r_instances = class_instances.get(r.iri, [])
                scope_instances.extend(r_instances)
            scope_instances = list(set(scope_instances))  # remove duplicates
        #append the result in csr_list and the offset in offsets
            valid_ids = [triples.entity_to_id[e] for e in scope_instances if e in triples.entity_to_id]
            csr_list.extend(valid_ids)
            current_offset += len(valid_ids)
            offsets.append(current_offset)
        
        return torch.tensor(csr_list), torch.tensor(offsets)
    

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
        hr_counts['score'] = (np.log1p(hr_counts['count']) / np.log1p(hr_counts['max']))*(1/(1+(hr_counts['std']/(hr_counts['median']+1e-6))))  # Adding a small epsilon to avoid division by zero
        # log(count(r,t)+1) / log(MAX_t'(r,t')+1)*lambda(r) where lambda(r) = 1/(1 + std/median)
        rt_counts['score'] = (np.log1p(rt_counts['count']) / np.log1p(rt_counts['max']))*(1/(1+(rt_counts['std']/(rt_counts['median']+1e-6))))  # Adding a small epsilon to avoid division by zero

        # 6. Rebuild the final dictionaries: {(h, r): score} and {(r, t): score}
        hr_scores = dict(zip(zip(hr_counts['h'], hr_counts['r']), hr_counts['score']))
        rt_scores = dict(zip(zip(rt_counts['r'], rt_counts['t']), rt_counts['score']))

        return hr_scores, rt_scores
    
    def build_graph(self, t_box) -> nx.DiGraph:
        """Builds the DAG from the t_box's subClassOf hierarchy."""
        G = nx.DiGraph()
        root = owlready2.owl.Thing.name
 
        for cls in t_box.classes():
            for parent in cls.is_a:
            # Filter Restrictions and other OWL non-class constructs
                if isinstance(parent, type) and issubclass(parent, owlready2.owl.Thing):
                    G.add_edge(parent.name, cls.name)
 
    # Attach classes without explicit parent to owl:Thing
        for cls in t_box.classes():
            if cls.name not in G.nodes:
                G.add_edge(root, cls.name)
            elif G.in_degree(cls.name) == 0 and cls.name != root:
                G.add_edge(root, cls.name)
        return G
 
 
    def compute_ic(self, G: nx.DiGraph) -> dict:
        """
        Computes the structural IC of Seco for each node in the DAG.
        IC(c) = 1 - log(|descendants(c)| + 1) / log(N)
        """
        N = G.number_of_nodes()
        ic = {}
        for node in G.nodes:
            hypo = len(nx.descendants(G, node))
            ic[node] = 1 - (math.log(hypo + 1) / math.log(N))
        return ic
 
 
    def get_lca(self, G: nx.DiGraph, ic: dict, c1: str, c2: str):
        """
        Finds the least common ancestor with the highest IC (the most specific).
        Returns None if no common ancestor exists.
        """
        anc1 = nx.ancestors(G, c1) | {c1}
        anc2 = nx.ancestors(G, c2) | {c2}
        common = anc1 & anc2
        if not common:
            return None
        return max(common, key=lambda n: ic.get(n, 0))
 
 
    def jiang_conrath_sim(self, ic: dict, G: nx.DiGraph, c1: str, c2: str) -> float:
        """
        Computes the Jiang & Conrath similarity normalized in [0, 1].

        dist(c1, c2) = IC(c1) + IC(c2) - 2 * IC(LCA)
        sim(c1, c2)  = 1 - dist / 2

        Limit cases:
        - Same class        → 1.0
        - Class not in graph → 0.0
        - No common LCA    → 0.0
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
 
 
    def build_similarity_matrix(self, t_box, relation_scope: dict) -> torch.Tensor:
        """
        Builds the r x r matrix of taxonomic similarity between the classes
        appearing in the scope of relations.

        Args:
            t_box:         TBox object
            relation_ranges:  dict {relation_id: class_name} mapping
                              each relation to its range class(es)

        Returns:
            Tensor (r x r) with sim[i, j] = taxonomic similarity between
            the range of relation i and the range of relation j.
        """
        G = self.build_graph(t_box)
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
        matrix.fill_diagonal_(0) # Set diagonal to 0 to avoid considering self-similarity       
        empty_mask = matrix.sum(dim=1) == 0  # If the row is all 0, the relation has no taxonomic similarity with any other relation
        matrix[empty_mask] = 1/r  # Set the row to 1/r to simulate random sampling when there is no taxonomic similarity
        
        return matrix, empty_mask
    
    def build_disjoint_with_matrix(self, t_box, relation_scope: dict) -> torch.Tensor:
        """Builds an r x r matrix of disjointness between the classes,
        returning 1 if the classes of two relations are disjoint and 0 otherwise.
        If the range is multiple, we return 0, as we cannot assert that all classes are disjoint with those of the other relation.
        """
        
        r=len(relation_scope)
        matrix = torch.full((r, r), 0.0)# i create a matrix of number sufficiently small

        #we precompute the dijointness, so that there are less calls to the ontology
        
        lookup = {}
        for i, classes in tqdm(relation_scope.items(), desc="Precomputing disjoint_relations"):
            if (len(classes) != 1 or 
                    classes[0] in [None, "None", "Thing", "owl:Thing"]): #there are multiscope properties, not yet supported, we leave this challenge to future brave researchers
                lookup[i] = (None, set())
            else:
                cls = t_box.search_one(iri=f'*{classes[0]}')
                disjoints=list(cls.disjoints())[0].entities if len(list(cls.disjoints()))>0 else []
                lookup[i] = (cls, set(disjoints))
                
        for i, (cls_i, disjoint_i) in lookup.items():
            for j, (cls_j, _) in lookup.items():
                if cls_i is None or cls_j is None:
                    continue
                else:
                    matrix[i, j] = 1.0 if cls_j in disjoint_i else 0

        empty_mask = matrix.sum(dim=1) == 0  # if the row is all 0, it means that the relation scope has no disjointness with any other relation scope
        matrix[empty_mask] = 1/r  # we set the row to 1/r, to simulate random sampling, as there is no disjointness with any other relation
        return matrix, empty_mask