import wandb
import torch

from agents import PolicyLearning, ConstantLearner

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def initialCharacterization():
    """ How does reproduction accuracy scale with number of images? """

    sweep_configuration = {
        'method': 'grid',
        'name': 'Initial initialCharacterization',
        'parameters':
        {
            'mu': {'values': [0.1, 0.2, 0.5]},
            'sigma': {'values': [0.1, 0.2, 0.5]},
            'utilityFn': {'values': ['linear', 'sqrt', 'log']},
         }
    }

    def initAndTrain():
        wandb.init(group='initialCharacterization')

        model = PolicyLearning(**wandb.config)
        targetdir = 'initialCharacterize/mu{mu}sigma{sigma}util{util}'.format(
                mu=wandb.config.mu, sigma=wandb.config.sigma,
                util=wandb.config.utilityFn
            )
        checkpoint_callback = ModelCheckpoint(
                            dirpath=f'lightning_logs/{targetdir}',
                            every_n_epochs=1
                            )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=1000,
            callbacks=[checkpoint_callback]
            )
        trainer.fit(model)

        wandb.finish()

    sweep_id = wandb.sweep(sweep_configuration, project='portfolioRL')
    wandb.agent(sweep_id, function=initAndTrain)


def basicTraining(model, dirname, **kwargs):
    wandb_logger = WandbLogger(project='portfolioRL')

    checkpoint_callback = ModelCheckpoint(dirpath=f'lightning_logs/{dirname}')
    trainer = Trainer(
                      logger=wandb_logger,
                      **kwargs,
                      callbacks=[checkpoint_callback]
                      )
    trainer.fit(model)

    wandb.finish()


def runSweep(cfg, modelClass, **trainerKwargs):
    groupName = cfg['name']

    def initAndTrain():
        wandb.init(group=groupName)

        model = modelClass(**wandb.config)

        targetdir = ''.join(
                [groupName] +
                ['-{key}{value}'.format(key=key, value=wandb.config[key])
                    for key in cfg['parameters']
                 ])

        checkpoint_callback = ModelCheckpoint(
                        dirpath=f'lightning_logs/{groupName}Sweep/{targetdir}',
                        every_n_epochs=1
                        )

        callbacks = trainerKwargs.get('callbacks', [])
        callbacks.append(checkpoint_callback)

        otherKwargs = {k: trainerKwargs[k]
                       for k in trainerKwargs if k != 'callbacks'}

        if 'max_epochs' not in trainerKwargs:
            trainerKwargs['max_epochs'] = 2000

        wandb_logger = WandbLogger()

        trainer = Trainer(
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=callbacks,
            **otherKwargs
            )
        trainer.fit(model)

        wandb.finish()

    sweep_id = wandb.sweep(cfg, project='portfolioRL')
    wandb.agent(sweep_id, function=initAndTrain)


""" Sweep definitions """
mean_sweep_0 = {
    'method': 'grid',
    'name': 'temp_name',
    'parameters':
    {
        'mu': {'values': torch.logspace(-3, -0.9, 12).tolist()},
        'sigma': {'values': [0.25]},
        'utilityFn': {'values': ['sqrt']},
     }
}

earlyStopping_callback = EarlyStopping(monitor='total certainty', 
                                       min_delta=-float('inf'),
                                       stopping_threshold=100., mode='max')
