import owlready2
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

from code.precomputing import build_graph, compute_ic, get_lca, jiang_conrath_sim, jiang_conrath_sim

class SchemaExtractor:
    def __init__(self, ontology_path, triples):
        
        self.ontology=owlready2.get_ontology(ontology_path).load()
        
        self.triples=triples
        
        self.class_instances=self.class_to_entity(self.ontology)
        
        self.relation_ranges=self.relation_to_range(self.ontology, self.triples, extract="range")
        
        self.relation_domains=self.relation_to_range(self.ontology, self.triples, extract="domain")
        
        self.csr_list_r, self.offsets_r=self.CSR_offset_precomputing(
            self.ontology,
            self.triples,
            self.class_instances,
            extract="range")
        
        self.csr_list_d, self.offsets_d=self.CSR_offset_precomputing(
            self.ontology,
            self.triples,
            self.class_instances,
            extract="domain")
        
        self.hr_scores, self.rt_scores=self.calcola_score_normalizzati(self.triples)
        
        self.similarity_matrix_r=self.build_similarity_matrix(self.ontology, self.relation_ranges)
        
        self.similarity_matrix_d=self.build_similarity_matrix(self.ontology, self.relation_domains)
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
            if hasattr(r, 'Classes'):  # è un Or, spacchettalo
                classes.extend(r.Classes)
            else:
                classes.append(r)
        return classes

    def relation_to_range(self, ontology, triples, extract: Literal["range", "domain"]):
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
            #gestiamo i casi in cui r sia un semplice range o una unionof di classi
        return relation_scope
        


#scansiono sulle relazioni e estraggo il domain o il range, facendo fede al dizioario classe-entita creo un tensore
# e un offset
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

    def calcola_score_normalizzati(self, triples):
    # 1. Estraiamo le colonne come liste Python (è l'operazione più veloce in assoluto)
        h_array = triples.mapped_triples[:, 0].tolist()
        r_array = triples.mapped_triples[:, 1].tolist()
        t_array = triples.mapped_triples[:, 2].tolist()

    # 2. Contiamo le frequenze assolute creando direttamente le tuple (h, r) e (r, t)
        print("Conteggio frequenze in corso...")
        hr_counts = Counter(zip(h_array, r_array))
        rt_counts = Counter(zip(r_array, t_array))

    # 3. Troviamo il MAX per ogni singola relazione r
    # Usiamo defaultdict così non dobbiamo inizializzare a 0 manualmente
        max_hr_for_r = defaultdict(int)
        max_rt_for_r = defaultdict(int)

        print("Calcolo dei massimi per relazione...")
        for (h, r), count in hr_counts.items():
            if count > max_hr_for_r[r]:
                max_hr_for_r[r] = count

        for (r, t), count in rt_counts.items():
            if count > max_rt_for_r[r]:
                max_rt_for_r[r] = count

    # 4. Creiamo i due dizionari finali con la divisione (score da 0 a 1)
        print("Generazione dei dizionari finali...")
        hr_scores = {}
        for (h, r), count in hr_counts.items():
        # count(h,r) / MAX_h'(h', r)
            hr_scores[(h, r)] = count / max_hr_for_r[r]

        rt_scores = {}
        for (r, t), count in rt_counts.items():
        # count(r,t) / MAX_t'(r, t')
            rt_scores[(r, t)] = count / max_rt_for_r[r]

        print("Completato!")
        return hr_scores, rt_scores
    
    def build_graph(self, ontology) -> nx.DiGraph:
        """Costruisce il DAG dalla gerarchia subClassOf dell'ontologia."""
        G = nx.DiGraph()
        root = owlready2.owl.Thing.name
 
        for cls in ontology.classes():
            for parent in cls.is_a:
            # Filtra Restriction e altri costrutti OWL non-classe
                if isinstance(parent, type) and issubclass(parent, owlready2.owl.Thing):
                    G.add_edge(parent.name, cls.name)
 
    # Aggancia classi senza parent esplicito a owl:Thing
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
        - Stessa classe        → 1.0
        - Classe non nel grafo → 0.0
        - Nessun LCA comune    → 0.0
        """
        if c1 not in ic or c2 not in ic or c1==c2:
            return 0.0
        lca = self.get_lca(G, ic, c1, c2)
        if lca is None:
            return 0.0
        dist = ic[c1] + ic[c2] - 2 * ic[lca]
        return 1.0 - (dist / 2.0)
 
 
    def build_similarity_matrix(self, ontology, relation_ranges: dict) -> torch.Tensor:
        """
        Costruisce la matrice r x r di similarità tassonomica tra le classi
        che compaiono nei range delle relazioni.
 
        Args:
            ontology:         oggetto ontologia OWLready2
            relation_ranges:  dict {relation_id: class_name} che mappa
                          ogni relazione alla sua classe di range
 
        Returns:
            Tensore (r x r) con sim[i, j] = similarità tassonomica tra
            il range di relazione i e il range di relazione j.
        """
        G = self.build_graph(ontology)
        ic = self.compute_ic(G)
    
        r = len(relation_ranges)
        matrix = torch.zeros(r, r)
 
        for i, classes_i in relation_ranges.items():
            for j, classes_j in relation_ranges.items():
                scores = [
                    self.jiang_conrath_sim(ic, G, ci, cj)
                    for ci in classes_i
                    for cj in classes_j
                ]
                matrix[i, j] = sum(scores) / (len(scores) + 1e-6)
        return matrix

        '''
        TODO:
        - QUANDO UNA RELAZIONE NON HA RANGE METTI LE RIGHE DELLA MATRICE A 1 PER SIMULARE IL CAMPIONAMENTO CASUALE
        - MODELLARE IL CASO NON RANGE ANCHE NEGLI OFFSET E NELLA CSR, IN MODO DA AVERE UNA RAPPRESENTAZIONE UNIFORME PER TUTTE LE RELAZIONI
        
        '''