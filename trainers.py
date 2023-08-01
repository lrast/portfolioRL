import wandb
import torch

from agents import PolicyLearning

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


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


def runSweep(cfg):
    groupName = cfg['name']

    def initAndTrain():
        wandb.init(group=groupName)

        model = PolicyLearning(**wandb.config)
        targetdir = ''.join(
                [groupName] +
                ['-{key}{value}'.format(key=key, value=wandb.config[key])
                    for key in cfg['parameters']
                 ])

        checkpoint_callback = ModelCheckpoint(
                            dirpath=f'lightning_logs/{targetdir}',
                            every_n_epochs=1
                            )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            logger=wandb_logger,
            log_every_n_steps=1,
            max_epochs=2000,
            callbacks=[checkpoint_callback]
            )
        trainer.fit(model)

        wandb.finish()

    sweep_id = wandb.sweep(cfg, project='portfolioRL')
    wandb.agent(sweep_id, function=initAndTrain)


""" Sweep definitions """
mean_sweep_0 = {
    'method': 'grid',
    'name': 'dynamicSweep_mean',
    'parameters':
    {
        'mu': {'values': torch.logspace(-3, -0.9, 12).tolist()},
        'sigma': {'values': [0.25]},
        'utilityFn': {'values': ['sqrt']},
     }
}
