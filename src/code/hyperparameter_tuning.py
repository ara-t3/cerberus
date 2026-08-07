
from pykeen.regularizers import LpRegularizer
from pykeen.models import TransE, RotatE, TuckER
from pykeen.optimizers import Adam
from pykeen.losses import MarginRankingLoss, BCEAfterSigmoidLoss
from pykeen.training import SLCWATrainingLoop
from pykeen.sampling import BernoulliNegativeSampler, PseudoTypedNegativeSampler
from pykeen.evaluation import RankBasedEvaluator
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline, HpoPipelineResult
from pykeen.triples import TriplesFactory
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage
from pykeen.stoppers import EarlyStopper
from pykeen.checkpoints import save_model
import torch
import optuna
import json
import argparse
import os
from pathlib import Path
from src.code.FNWeighter import FNLossWeighter
from src.code.SchemaNegativeSampling import SchemaNegativeSampler
from pykeen.sampling import negative_sampler_resolver

negative_sampler_resolver.register(SchemaNegativeSampler, ["schema_sampler"])

parser = argparse.ArgumentParser()
parser.add_argument("--g", required=True)
parser.add_argument("--model", type=str, default='transe', 
                    choices=['transe', 'rotate', 'complex'], 
                    help="Modello da ottimizzare")
parser.add_argument("--sampler", type=str, default='schema',
                    choices=['bernoulli', 'schema', 'schema_ablation'],
                    help="Negative sampler to use")
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
SAMPLER=args.sampler
TRIPLES=f'{PROJECT_ROOT}/data/{args.g}_RON/abox/splits'
ABOX_PATH=f'{PROJECT_ROOT}/data/{args.g}_RON/abox/class_assertions.owl'
TBOX_PATH=f'{PROJECT_ROOT}/data/{args.g}_RON/ontology.owl'
MODEL = args.model.lower() 

os.makedirs(f'optuna_checkpoints/{args.g}', exist_ok=True)
os.makedirs(f'hpo_results/{args.g}', exist_ok=True)
storage = RDBStorage(f"sqlite:///optuna_checkpoints/{args.g}/{MODEL}_{SAMPLER}_optuna_study.db")

train = TriplesFactory.from_path(
    path=os.path.join(TRIPLES,'train.tsv'),
    )

val = TriplesFactory.from_path(
        path=os.path.join(TRIPLES,'valid.tsv'),
        entity_to_id=train.entity_to_id,
        relation_to_id=train.relation_to_id,
    )

test = TriplesFactory.from_path(
        path=os.path.join(TRIPLES,'test.tsv'),
        entity_to_id=train.entity_to_id,
        relation_to_id=train.relation_to_id,
    )

model_configs = {
    'transe': {
        'model_kwargs': dict(
            embedding_dim=512,
            scoring_fct_norm=1, 
        )
    },
    'rotate': {
        'model_kwargs': dict(
            embedding_dim=512,
        )
    },
    'complex': {
        'model_kwargs': dict(
            embedding_dim=512,
        )
    }
}


loss_weighter=None
current_config = model_configs[MODEL]


if SAMPLER == 'schema':
        loss_weighter=FNLossWeighter()
        negative_sampler = 'schema_sampler'
        negative_sampler_kwargs = dict(
            filtered=True,
            a_box=ABOX_PATH,
            t_box=TBOX_PATH,
            triples_factory=train,
            loss_weighter=loss_weighter
        )

elif SAMPLER == 'bernoulli':
        negative_sampler = BernoulliNegativeSampler
        negative_sampler_kwargs = dict(
            filtered=True,
        )
        
elif SAMPLER == 'schema_ablation':
        negative_sampler = SchemaNegativeSampler
        negative_sampler_kwargs = dict(
            filtered=True,
            a_box=ABOX_PATH,
            t_box=TBOX_PATH,
            triples_factory=train,
            loss_weighter=None
        )

hpo_kwargs = dict(
    storage=storage,
    study_name=f'{MODEL}_{SAMPLER}_hpo_study',
    load_if_exists=True,
    n_trials=10,
    training_kwargs=dict(num_epochs=200, batch_size=512),
    training=train,
    validation=val,
    testing=test,
    model=MODEL,
    
    #stopper='early',
    #stopper_kwargs=dict(frequency=25, patience=3, relative_delta=0.002),

    optimizer='adam',
    optimizer_kwargs_ranges=dict(lr=dict(type='categorical', choices=[1e-4, 1e-3, 1e-2])),

    loss='NSSALoss',
    loss_kwargs_ranges=dict(
        margin=dict(type='categorical', choices=[3, 9, 18]), 
        adversarial_temperature=dict(type=float, low=0.5, high=1.0), #.
    ),

    training_loop='slcwa',
    training_loop_kwargs=dict(
        loss_weighter=loss_weighter,
    ),
    regularizer='lpregularizer',
    regularizer_kwargs_ranges=dict(
        p=dict(type=int, low=1, high=2),
        weight=dict(type='categorical', choices=[1e-4, 1e-3, 1e-2])
    ),

    negative_sampler=negative_sampler,
    negative_sampler_kwargs=negative_sampler_kwargs,
    negative_sampler_kwargs_ranges=dict(
        num_negs_per_pos=dict(type='categorical', choices=[2, 10, 40])),

    evaluator='RankBasedEvaluator',
    evaluator_kwargs=dict(filtered=True),
    evaluation_kwargs=dict(batch_size=2000),
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=5
        ),
    
    device='cuda'
)

hpo_kwargs.update(current_config)

optimization_result = hpo_pipeline(**hpo_kwargs)

out_dir = f'hpo_results/{args.g}/{MODEL}_{SAMPLER}_hpo_results'
os.makedirs(out_dir, exist_ok=True)

best_params = optimization_result.study.best_params

with open(os.path.join(out_dir, 'best_config.json'), 'w') as f:
    json.dump(best_params, f, indent=4)

print('study_saved')

