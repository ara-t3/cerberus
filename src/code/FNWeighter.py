from pykeen.triples.weights import LossWeighter

class FNLossWeighter(LossWeighter):
    def __init__(self):
        super().__init__()
        self.batch_scores=None
                
    def __call__(self, mapped_triples):
        super().__call__(mapped_triples)

    def weight_triples(self, mapped_triples):
        
        if self.batch_scores is None:
            return None
        else:
            negative_weights = self.batch_scores
            #print(negative_weights)
            return negative_weights.to(mapped_triples.device)
        
        
if __name__ == "__main__":
    mario=FNLossWeighter('ciao')
    