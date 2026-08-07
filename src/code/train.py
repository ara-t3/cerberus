
from pykeen.triples import TriplesFactory
from src.code.Schema import SchemaTranslator
from src.code.SchemaNegativeSampling import SchemaNegativeSampler
from src.code.FNWeighter import FNLossWeighter
import os
from pykeen.pipeline import pipeline
from pykeen.losses import MarginRankingLoss, NSSALoss
from pykeen.regularizers import LpRegularizer
from pykeen.sampling import BernoulliNegativeSampler, BasicNegativeSampler, PseudoTypedNegativeSampler
from pykeen.stoppers import EarlyStopper
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
import torch
import json
import argparse
import os
import optuna
from pathlib import Path



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--g", required=True)
    parser.add_argument("--model", type=str, default='transe', 
                        choices=['transe', 'rotate', 'complex']),
    parser.add_argument("--sampler", type=str, default='schema',
                        choices=['bernoulli', 'schema', 'schema_ablation'],
                        help="Negative sampler to use")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    MODEL = args.model.lower()  # Options: 'transe', 'rotate', 'tucker'
    SAMPLER=args.sampler  # Options: 'BERNOULLI', 'RANDOM', 'PSEUDOTYPED', 'SCHEMA'
    TRIPLES=f'{PROJECT_ROOT}/data/{args.g}_RON/abox/splits'
    ABOX_PATH=f'{PROJECT_ROOT}/data/{args.g}_RON/abox/class_assertions.owl'
    TBOX_PATH=f'{PROJECT_ROOT}/data/{args.g}_RON/ontology.owl'
    OPTUNA_STUDY=f'{PROJECT_ROOT}/hpo_results/{args.g}/{MODEL}_{SAMPLER}_hpo_results'
    
    dirs = ['results', 'pipelines', 'losses', 'checkpoints']

    for dir in dirs:
        os.makedirs(f'{dir}/{args.g}', exist_ok=True)
        
    
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
    
    best_params=optuna.load_study(study_name=f'{MODEL}_{SAMPLER}_hpo_study',
                                   storage=f"sqlite:///optuna_checkpoints/{args.g}/{MODEL}_{SAMPLER}_optuna_study.db")

    # Modello
    EMBEDDING_DIM = 256
    SCORING_FCT_NORM = 1         
    ENTITY_INITIALIZER = 'xavier_uniform'

    # Loss
    MARGIN = best_params.best_params['loss.margin']

    # Regularizer
    REGULARIZER_P = best_params.best_params['regularizer.p'] 
    REGULARIZER_WEIGHT = best_params.best_params['regularizer.weight']

    # Optimizer
    LEARNING_RATE = best_params.best_params['optimizer.lr']

    # Negative sampler
    NUM_NEGS_PER_POS = best_params.best_params['negative_sampler.num_negs_per_pos']
    
    LOSS_ADVERSARIAL_TEMPERATURE = best_params.best_params['loss.adversarial_temperature']
    
    loss_weighter=None
    
    if SAMPLER == 'bernoulli':
        negative_sampler = BernoulliNegativeSampler
        negative_sampler_kwargs = dict(
            num_negs_per_pos=NUM_NEGS_PER_POS,
            filtered=True,
        )
    
    elif SAMPLER == 'schema':
        loss_weighter=FNLossWeighter()
        negative_sampler = SchemaNegativeSampler
        negative_sampler_kwargs = dict(
            num_negs_per_pos=NUM_NEGS_PER_POS,
            filtered=True,
            a_box=ABOX_PATH,
            t_box=TBOX_PATH,
            triples_factory=train,
            loss_weighter=loss_weighter
        )
    elif SAMPLER == 'schema_ablation':
        negative_sampler = SchemaNegativeSampler
        negative_sampler_kwargs = dict(
            num_negs_per_pos=NUM_NEGS_PER_POS,
            filtered=True,
            a_box=ABOX_PATH,
            t_box=TBOX_PATH,
            triples_factory=train,
            loss_weighter=None
        )
    else:
        raise ValueError(f'Unknown sampler: {SAMPLER}')

    # Training loop
    NUM_EPOCHS = 200
    BATCH_SIZE = 512

    # Early stopping
    STOPPER_FREQUENCY = 25
    STOPPER_PATIENCE = 3
    STOPPER_RELATIVE_DELTA = 0.002

    # Evaluation
    EVAL_BATCH_SIZE = 256

    # ============================================================
    # TRAINING PIPELINE
    # ============================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    
    
    
    model=args.model
    if model == 'transe':
        model_kwargs=dict(
            embedding_dim=EMBEDDING_DIM,
            scoring_fct_norm=SCORING_FCT_NORM,
            entity_initializer=ENTITY_INITIALIZER,
        )
    else:
        model_kwargs=dict(
            embedding_dim=EMBEDDING_DIM,
            entity_initializer=ENTITY_INITIALIZER,
        )
    
    result = pipeline(
        training=train,
        validation=val,
        testing=test,

        # --- Model ---
        model=model,
        model_kwargs=model_kwargs,

        # --- Loss ---
        loss=NSSALoss(),
        loss_kwargs=dict(
            margin=MARGIN,
            adversarial_temperature=LOSS_ADVERSARIAL_TEMPERATURE,
        ),
    

        # --- Regularizer ---
        regularizer=LpRegularizer,
        regularizer_kwargs=dict(
            p=REGULARIZER_P,
            weight=REGULARIZER_WEIGHT,
        ),

        # --- Optimizer ---
        optimizer='Adam',
        optimizer_kwargs=dict(
            lr=LEARNING_RATE,
        ),

        # --- Negative sampler ---
        negative_sampler=negative_sampler,
        negative_sampler_kwargs=negative_sampler_kwargs,
        # --- Training loop ---
        training_loop='sLCWA',
        training_loop_kwargs=dict(
            loss_weighter=loss_weighter,
        ),
        training_kwargs=dict(
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            checkpoint_name=f'{model}_checkpoint_{SAMPLER}.pt',
            checkpoint_frequency=0,
            checkpoint_directory=f'checkpoints/{args.g}',
        ),

        # --- Early stopping ---
        stopper='early',
        stopper_kwargs=dict(
            frequency=STOPPER_FREQUENCY,
            patience=STOPPER_PATIENCE,
            relative_delta=STOPPER_RELATIVE_DELTA,
            metric='mean_reciprocal_rank',
            larger_is_better=True,
        ),

        # --- Evaluation ---
        evaluator=RankBasedEvaluator,
        evaluator_kwargs=dict(
            filtered=True,
        ),
        evaluation_kwargs=dict(
            batch_size=EVAL_BATCH_SIZE,
        ),

        # --- Varie ---
        random_seed=43,
        device=device,
    )

    # --- Salvataggio modello e loss ---
    model = result.model

    losses = {'loss': result.losses}
    with open(f'losses/{args.g}/{model}_{SAMPLER}.json', 'w') as f:
        json.dump(losses, f)

    result.save_to_directory(f'pipelines/{args.g}/{model}_{SAMPLER}_pipeline_results')
    print('****** MODEL SAVED ******')

    # --- Metriche ---
    metric_result = result.metric_results

    print(f"Hits@1: \t\t{metric_result.get_metric('hits@1'): 5.3f}")
    print(f"Hits@3: \t\t{metric_result.get_metric('hits@3'): 5.3f}")
    print(f"Hits@5: \t\t{metric_result.get_metric('hits@5'): 5.3f}")
    print(f"Hits@10: \t\t{metric_result.get_metric('hits@10'): 5.3f}")
    print(f"Mean Reciprocal Rank:  \t{metric_result.get_metric('mean_reciprocal_rank'): 5.3f}")
    print(f"Mean Rank:  \t\t{metric_result.get_metric('mean_rank'): 5.3f}")

    with open(f'results/{args.g}/results_{model}_{SAMPLER}.json', 'w') as f:
        json.dump(metric_result.to_dict(), f)

    print('****** RESULTS SAVED ******')