import os
import inspect
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy, SketchyRetrieval
from experiments.options import opts

if __name__ == '__main__':
    seed_everything(opts.seed, workers=True)

    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    evaluation_categories = Sketchy.categories_for_mode(opts, mode='val')
    query_dataset = SketchyRetrieval(
        opts, dataset_transforms, evaluation_categories, modality='sketch')
    gallery_dataset = SketchyRetrieval(
        opts, dataset_transforms, evaluation_categories, modality='photo')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.workers,
        shuffle=True)
    query_loader = DataLoader(
        dataset=query_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.workers,
        shuffle=False)
    gallery_loader = DataLoader(
        dataset=gallery_dataset,
        batch_size=opts.batch_size,
        num_workers=opts.workers,
        shuffle=False)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models/%s'%opts.exp_name,
        filename="{epoch:02d}",
        save_top_k=0,
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print ('resuming training from %s'%ckpt_path)

    trainer_kwargs = dict(
        min_epochs=1,
        max_epochs=opts.max_epochs,
        benchmark=False,
        deterministic=True,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )

    trainer_init_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_fit_parameters = inspect.signature(Trainer.fit).parameters
    if 'gpus' in trainer_init_parameters:
        trainer_kwargs['gpus'] = -1
        if 'ckpt_path' not in trainer_fit_parameters:
            trainer_kwargs['resume_from_checkpoint'] = ckpt_path
    else:
        trainer_kwargs['accelerator'] = 'auto'
        trainer_kwargs['devices'] = 'auto'

    trainer = Trainer(**trainer_kwargs)
    model = Model(categories=train_dataset.all_categories)

    print ('beginning training...good luck...')
    fit_kwargs = {}
    if 'ckpt_path' in trainer_fit_parameters:
        fit_kwargs['ckpt_path'] = ckpt_path
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=[query_loader, gallery_loader],
        **fit_kwargs)
