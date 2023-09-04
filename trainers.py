import wandb
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


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
        'mu': {'values': torch.logspace(-2.5, -0.8, 8).tolist()},
        'sigma': {'values': [0.25]},
     }
}
