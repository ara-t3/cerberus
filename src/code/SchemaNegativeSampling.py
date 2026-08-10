from pykeen.sampling import BernoulliNegativeSampler
import torch
from typing import Literal
from src.code.Schema import SchemaTranslator
import warnings


class SchemaNegativeSampler(BernoulliNegativeSampler):
    def __init__(self, num_negs_per_pos: int = 1, t_box=None, a_box=None, loss_weighter=None, **kwargs):
        
        triples = kwargs.pop('triples_factory') or self
        mapped_triples = kwargs.pop('mapped_triples')
        
        self.loss_weighter = loss_weighter
        if self.loss_weighter is None:
            warnings.warn('''
                            \n[WARNING] No 'weighter' was passed to SchemaNegativeSampler.
\nThe ontological scoring system will not be active! potential false negatives will NOT be weighted down!
To enable it, instantiate 'SchemaLossWeighter()' upstream and pass it both in
'negative_sampler_kwargs' (as loss_weighter=...) and in 'training_loop_kwargs' (as loss_weighter=...).
                          ''')
        
        super().__init__(mapped_triples=mapped_triples, num_negs_per_pos=num_negs_per_pos, **kwargs)
        self.schema_extractor = SchemaTranslator(t_box=t_box, a_box=a_box, triples=triples)
        
    def __extract_negatives(self, batch_size, r, empty_mask, num_negs_per_pos, csr, offsets, true_entities, device):
        # 2. Starts and lengths
        starts = offsets[r] #SHAPE: (batch_size * num_negs_per_pos,)
        ends = offsets[r + 1] #SHAPE: (batch_size * num_negs_per_pos,)
        lengths = ends - starts
        # Identify where the length is 0 (missing range)
        axiom_failure_mask = (lengths == 0) | empty_mask  #SHAPE: (batch_size * num_negs_per_pos,)
        
        # Avoid math issues by forcing lengths to 1 where they are 0
        # (we will overwrite these values afterwards)
        safe_lengths = torch.where(axiom_failure_mask, torch.ones_like(lengths), lengths) #SHAPE: (batch_size * num_negs_per_pos,)

        # 3. Roll the dice and get IDs from the range
        u = torch.rand(batch_size*num_negs_per_pos, device=device)
        negative_csr_ids = starts + (u * safe_lengths).long() #SHAPE: (batch_size * num_negs_per_pos,)
        negative_csr_ids = torch.clamp(negative_csr_ids, max=ends-1) #SHAPE: (batch_size * num_negs_per_pos,)

        # 4. Gather from the CSR tensor
        negative_entities = csr[negative_csr_ids] #shape: (batch_size * num_negs_per_pos,)
        
        # --- FALLBACK FOR EMPTY RANGE ---
        # Since we cannot return a smaller batch than the input,
        # return the positive triple as negative (which is certainly a false negative)
        # in cases where the range is empty.
        # Replace IDs only where the mask indicates the range was empty
        negative_entities = torch.where(axiom_failure_mask, true_entities, negative_entities)
    
        negative_entities = negative_entities.view(batch_size, num_negs_per_pos) #shape: (batch_size, num_negs_per_pos)
        axiom_failure_mask = axiom_failure_mask.view(batch_size, num_negs_per_pos)
        

        
        return negative_entities, axiom_failure_mask

    def __corrupt_triple_inside_scope(self, positive_batch, num_negs_per_pos, csr, offsets, scores, empty_mask, corruption: Literal["head", "tail"] = "tail"):
        batch_size = positive_batch.shape[0]
        
        if corruption == "tail":
            idx0, idx1 = positive_batch[:, 0], positive_batch[:, 1]  # (h, r)
        else:
            idx0, idx1 = positive_batch[:, 1], positive_batch[:, 2]  # (r, t)
        batch_scores = scores[idx0, idx1]
        
        batch_scores=batch_scores.unsqueeze(1).expand(-1, num_negs_per_pos).view(batch_size, num_negs_per_pos) #shape: (batch_size, num_negs_per_pos)
        
        # 1. Create empty structure 
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, num_negs_per_pos, 1)
        r = positive_batch[:, 1].repeat_interleave(num_negs_per_pos) #shape: (batch_size * num_negs_per_pos,)
        true_entities = positive_batch[:, 2] if corruption == "tail" else positive_batch[:, 0] #shape: (batch_size,)
        true_entities = true_entities.repeat_interleave(num_negs_per_pos) #shape: (batch_size * num_negs_per_pos,)
        # 2-4. Extract negatives
        negative_ids, empty_axiom_mask = self.__extract_negatives(batch_size,
                                    r,
                                    empty_mask,
                                    num_negs_per_pos,
                                    csr,
                                    offsets,
                                    true_entities,
                                    positive_batch.device)

        # 5. Write into the negative batch
        negative_batch[
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            torch.arange(num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            2 if corruption == "tail" else 0
        ] = negative_ids
        
        batch_scores = torch.where(empty_axiom_mask, 1.0, batch_scores)
        
        return negative_batch, empty_axiom_mask, batch_scores

    def __corrupt_triple_outside_scope(self, positive_batch, similarity_matrix, empty_mask, num_negs_per_pos, csr, offsets, corruption: Literal["head", "tail"] = "tail"):
        batch_size = positive_batch.shape[0]
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, num_negs_per_pos, 1)
    
        r = positive_batch[:, 1] #shape: (batch_size,)
        empty_mask = empty_mask[r] #shape: (batch_size,)
        empty_mask = empty_mask.repeat_interleave(num_negs_per_pos) #shape: (batch_size * num_negs_per_pos,)
        batch_matrix=similarity_matrix[r] #shape: (batch_size, r)
        r=torch.multinomial(batch_matrix, num_samples=num_negs_per_pos, replacement=True) #shape: (batch_size, num_negs_per_pos)
        r=r.view(-1) #shape: (batch_size * num_negs_per_pos,)
        true_entities = positive_batch[:, 2] if corruption == "tail" else positive_batch[:, 0] #shape: (batch_size,)
        true_entities = true_entities.repeat_interleave(num_negs_per_pos) #shape: (batch_size * num_negs_per_pos,)
        # 2-4. Extract negatives
        negative_ids, empty_axiom_mask = self.__extract_negatives(batch_size,
                                    r,
                                    empty_mask,
                                    num_negs_per_pos,
                                    csr,
                                    offsets,
                                    true_entities,
                                    positive_batch.device)

        # 5. Write into the negative batch
        negative_batch[
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            torch.arange(num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            2 if corruption == "tail" else 0
        ] = negative_ids
        
        batch_scores=torch.ones(batch_size, num_negs_per_pos, device=positive_batch.device) #shape: (batch_size, num_negs_per_pos)

        return negative_batch, empty_axiom_mask, batch_scores

    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:
        """
        Corrupts a batch of positive triples to generate negative samples.

        Args:
            positive_batch (torch.LongTensor): A tensor of shape (batch_size, 3) containing the positive triples.

        Returns:
            torch.LongTensor: A tensor of shape (batch_size, num_negs_per_pos, 3) containing the negative samples.
        """
        
        batch_size = positive_batch.shape[0]
        # Get the device of the input batch
        cpu_device = positive_batch.device # pykeen dataloader always uses cpu
        cuda_device = self.schema_extractor.csr_list_d.device # Prende CUDA in automatico
        

        # Get the number of negatives per positive triple
        num_negs_per_pos = self.num_negs_per_pos

        head_corruption_probability = self.corrupt_head_probability[positive_batch[..., 1]].to(cuda_device)
        head_mask = torch.rand(
            *positive_batch.shape[:-1], device=cuda_device
        ) < head_corruption_probability.to(device=cuda_device)
        
        positive_batch = positive_batch.to(cuda_device)


        #corrupt head triples
        negative_batch_0_h, empty_mask_0_h, batch_scores_0_h = self.__corrupt_triple_inside_scope(
            positive_batch,
            num_negs_per_pos,
            self.schema_extractor.csr_list_d,
            self.schema_extractor.offsets_d,
            self.schema_extractor.rt_scores,
            empty_mask = torch.zeros(
                batch_size * num_negs_per_pos,
                dtype=torch.bool,
                device=cuda_device
            ), #a mask with all false, since we dont need the mask for abscence of disjointness, as we are inside the scope
            corruption="head"
        )
        
        negative_batch_0_t, empty_mask_0_t, batch_scores_0_t = self.__corrupt_triple_inside_scope(
            positive_batch,
            num_negs_per_pos,
            self.schema_extractor.csr_list_r,
            self.schema_extractor.offsets_r,
            self.schema_extractor.hr_scores,
            empty_mask = torch.zeros(
                batch_size * num_negs_per_pos,
                dtype=torch.bool,
                device=cuda_device), #a mask with all false, since we dont need the mask for abscence of disjointness, as we are inside the scope
            corruption="tail"
        )
        
        negative_batch_1_h, empty_mask_1_h, batch_scores_1_h = self.__corrupt_triple_outside_scope(
            positive_batch,
            self.schema_extractor.similarity_matrix_d,
            self.schema_extractor.empty_mask_d,
            num_negs_per_pos,
            self.schema_extractor.csr_list_d,
            self.schema_extractor.offsets_d,
            corruption="head"
        )
        
        negative_batch_1_t, empty_mask_1_t, batch_scores_1_t = self.__corrupt_triple_outside_scope(
            positive_batch,
            self.schema_extractor.similarity_matrix_r,
            self.schema_extractor.empty_mask_r,
            num_negs_per_pos,
            self.schema_extractor.csr_list_r,
            self.schema_extractor.offsets_r,
            corruption="tail"
        )
        
        
        #disjoints corruption
        negative_batch_3_h, empty_mask_3_h, batch_scores_3_h = self.__corrupt_triple_outside_scope(
            positive_batch,
            self.schema_extractor.disjoint_with_matrix_d,
            self.schema_extractor.empty_mask_disjoint_d,
            num_negs_per_pos,
            self.schema_extractor.csr_list_d,
            self.schema_extractor.offsets_d,
            corruption="head"
        )
        
        negative_batch_3_t, empty_mask_3_t, batch_scores_3_t = self.__corrupt_triple_outside_scope(
            positive_batch,
            self.schema_extractor.disjoint_with_matrix_r,
            self.schema_extractor.empty_mask_disjoint_r,
            num_negs_per_pos,
            self.schema_extractor.csr_list_r,
            self.schema_extractor.offsets_r,
            corruption="tail"
        )
        
        # now i put together the negative batches and the empty masks, like that: negative_batch_0_h, negative_batch_1_t, negative_batch_2_t, same for masks and scores
        negative_batch_h = torch.cat([negative_batch_0_h, negative_batch_1_h, negative_batch_3_h], dim=1)
        negative_batch_t = torch.cat([negative_batch_0_t, negative_batch_1_t, negative_batch_3_t], dim=1)
        empty_mask_h = torch.cat([empty_mask_0_h, empty_mask_1_h, empty_mask_3_h], dim=1)
        empty_mask_t = torch.cat([empty_mask_0_t, empty_mask_1_t, empty_mask_3_t], dim=1)
        batch_scores_h = torch.cat([batch_scores_0_h, batch_scores_1_h, batch_scores_3_h], dim=1)
        batch_scores_t = torch.cat([batch_scores_0_t, batch_scores_1_t, batch_scores_3_t], dim=1)
        
        #now i obtain only a single combined negative batch and a single combined empty mask, by selecting the negatives based on the head mask
        negative_batch = torch.where(head_mask[:, None, None], negative_batch_h, negative_batch_t)
        empty_mask = torch.where(head_mask[:, None], empty_mask_h, empty_mask_t)
        batch_scores = torch.where(head_mask[:, None], batch_scores_h, batch_scores_t)
        
        weights = (~empty_mask).float()
        
        no_axioms = weights.sum(dim=1) == 0
        weights[no_axioms] = 1.0
        
        sampled_ids = torch.multinomial(
                weights,
                num_samples=num_negs_per_pos,
                replacement=False,
        )
        
        sample_ids = sampled_ids.unsqueeze(-1).expand(-1, -1, 3)
        negative_batch = torch.gather(negative_batch, dim=1, index=sample_ids)
        batch_scores = torch.gather(batch_scores, dim=1, index=sampled_ids)

        if self.loss_weighter is not None:
            self.loss_weighter.batch_scores = batch_scores.to(cpu_device)

        return negative_batch.to(cpu_device)
