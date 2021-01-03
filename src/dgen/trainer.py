import os
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
from dgen.model import BaseModel
from dgen.data import ImageNetData
from pytorch_lightning.loggers import TestTubeLogger, CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    #if args.gpus and args.gpus > 1:
    #    args.distributed_backend='ddp'

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    if args.ckpt:
        model = BaseModel.load_from_checkpoint(args.ckpt)
    else:
        model = BaseModel(**vars(args))

    working_dir = args.default_root_dir
    if args.ckpt:
        exp_dir = os.path.dirname(args.ckpt).replace("checkpoints", "")
        args.id = "eval"
        working_dir = exp_dir
    logger = CSVLogger(working_dir, name=args.id, version=args.ver)
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(save_last=True,  verbose=True)

    trainer = pl.Trainer.from_argparse_args(args, logger=logger ,
                                            checkpoint_callback=checkpoint_callback)
    data_module = ImageNetData(args)

    if args.evaluate:
        trainer.test(model, datamodule=data_module)
    else:
        trainer.fit(model, data_module)
        #trainer.distributed_backend = 'dp'
        #trainer.test(model, datamodule=data_module)

def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--data-path', metavar='DIR', type=str,
                               help='path to dataset')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('--seed', type=int, default=1,
                               help='seed for initializing training.')
    parent_parser.add_argument('--id', type=str, default='test-exp',
                               help='exp id')
    parent_parser.add_argument('--ckpt', type=str, default=None,
                               help='a checkpoint to load (for evaluation)')

    parent_parser.add_argument('--ver', type=str, default=None,
                               help='exp version')


    parser = BaseModel.add_model_specific_args(parent_parser)
    parser = ImageNetData.add_data_specific_args(parser)

    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=90,
        default_root_dir="/misc/lmbraid19/saikiat/nets/robust/"
    )
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()
