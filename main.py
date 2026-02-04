import pickle
from os import makedirs, path, remove

import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision("high")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from src import utils
from src.centroids import get_centroid_preds
from src.config import read_args
from src.datamodule import ImageAuthorshipDataModule
from src.test import test_tripadvisor_authorship_task

from src.callbacks import EmissionsTrackerCallback

from pathlib import Path
import json
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

args = read_args()
pl.seed_everything(args.seed, workers=True)

if __name__ == "__main__":
    city = args.city
    workers = args.workers
    logger = None

    print("=" * 50)
    print(f"============= {city} ===========")

    model0 = args.model[0]

    use_tv = bool(getattr(args, "use_train_val", False))
    no_val = bool(getattr(args, "no_validation", False))
    use_rho = not bool(getattr(args, "ds_no_rho", False))   # por si ds_no_rho también viene None

    run_id = (
        f"d{args.d}__lr{args.lr}__do{args.dropout}__bs{args.batch_size}"
        f"__K{args.max_user_images}"
        f"__tv{int(use_tv)}__noval{int(no_val)}"
        f"__rho{int(use_rho)}__seed{args.seed}"
    )

    run_dir = Path("runs") / args.city / model0 / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # TensorBoard logger (siempre)
    logger = TensorBoardLogger(save_dir=str(run_dir / "tb"), name="", version="")

    # Guarda config del run
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # print(f"[RUN] {run_dir}")


    val_metric_name = (
        "val_loss" if args.model[0] not in ["PRESLEY", "COLLEI"] else "val_auc"
    )

    val_metric_mode = "min" if args.model[0] not in ["PRESLEY", "COLLEI"] else "max"

    # Initialize datamodule
    dm = ImageAuthorshipDataModule(
        city=city,
        batch_size=args.batch_size,
        num_workers=workers,
        dataset_class=utils.get_dataset_constructor(model0),
        use_train_val=args.use_train_val,
        limit_users=args.limit_users,
        model_name=model0,
        max_user_images=args.max_user_images,
        no_validation=args.no_validation,
    )

    # Initialize trainer
    if (not args.early_stopping) or args.no_validation:
        checkpointing = ModelCheckpoint(
            save_last=True,
            save_top_k=0,
            dirpath=str(run_dir / "checkpoints"),
            save_on_train_epoch_end=True,
        )
        callbacks = [checkpointing]
    else:
        early_stopping = EarlyStopping(
            monitor=val_metric_name,
            mode=val_metric_mode,
            min_delta=1e-4,
            patience=10,
            check_on_train_epoch_end=False,
        )
        checkpointing = ModelCheckpoint(
            save_top_k=1,
            monitor=val_metric_name,
            mode=val_metric_mode,
            dirpath=str(run_dir / "checkpoints"),
            filename="best",
            save_on_train_epoch_end=False,
        )
        callbacks = [checkpointing, early_stopping]


    callbacks.append(EmissionsTrackerCallback(log_to_trainer=True))
    callbacks.append(LearningRateMonitor(logging_interval="step"))


    # Optional CSV logging
    if args.log_to_csv:
        logger = pl.loggers.CSVLogger(
            name=city,
            version=f'{args.model[0]}{"_no_val" if args.no_validation else ""}{"_"+ args.logdir_name if args.logdir_name else ""}',
            save_dir="csv_logs",
        )

    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        accelerator="auto",
        strategy="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_model_summary=True,
        num_sanity_val_steps=0 if args.no_validation else 2,
    )

    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = args.limit_val_batches
    if args.limit_test_batches is not None:
        trainer_kwargs["limit_test_batches"] = args.limit_test_batches

    trainer = pl.Trainer(**trainer_kwargs)

    ### TRAIN MODE ###
    if args.stage == "train":
        dm.setup()
        model_name = args.model[0]
        model = utils.get_model(model_name, vars(args), dm.nusers)

        # Entrena SIEMPRE; si no_validation, Lightning simplemente no usará val si tu dm no lo da / o si limit_val_batches=0
        trainer.fit(model=model, datamodule=dm)


    ### HYPERPARAMETER TUNING MODE ###
    if args.stage == "tune":
        model_name = args.model[0]

        # Search space
        config = {
            "lr": 1e-3,
            "d": tune.choice([64, 128, 256, 512, 1024]),
            "dropout": tune.choice([0.5, 0.6, 0.7, 0.8, 0.9]),
        }

        # Report callback
        tunecallback = TuneReportCallback(
            metrics={
                "val_auc": "val_auc",
                "val_recall": "val_recall",
                "train_loss": "train_loss",
            },
            on=["validation_end", "train_end"],
        )

        # Command line reporter
        reporter = CLIReporter(parameter_columns=["d", "lr", "dropout"])

        # Basic function to train each one
        def train_config(config, datamodule=None):
            logger = pl.loggers.CSVLogger(
                name=city + "_tune/" + args.model[0],
                version="d_"
                + str(config["d"])
                + "_lr_"
                + str(config["lr"])
                + "_dropout_"
                + str(config["dropout"]),
                save_dir="C:/Users/Komi/Papers/BRIE/csv_logs",
            )

            trainer = pl.Trainer(
                max_epochs=75,
                accelerator="gpu",
                devices=[0],
                callbacks=[tunecallback, early_stopping],
                enable_progress_bar=False,
                logger=logger,
            )
            model = utils.get_model(model_name, config, nusers=datamodule.nusers)
            trainer.fit(
                model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )
            # Guardar la ruta del last.ckpt del run
            ckpt_last = run_dir / "checkpoints" / "last.ckpt"
            if ckpt_last.exists():
                with open(run_dir / "ckpt_path.txt", "w", encoding="utf-8") as f:
                    f.write(str(ckpt_last))


        # Execute analysis
        analysis = tune.run(
            tune.with_parameters(train_config, datamodule=dm),
            resources_per_trial={"cpu": 16, "gpu": 1},
            metric="val_auc",
            mode="max",
            config=config,
            num_samples=args.num_models,
            name=f"tune_{model_name}",
        )

        # Find best configuration and its best val metric value
        best_config = analysis.get_best_config(
            metric=val_metric_name, scope="all", mode=val_metric_mode
        )
        best_val_loss = analysis.dataframe(
            metric=val_metric_name, mode=val_metric_mode
        )[val_metric_name].max()

        print(f"Best {val_metric_name}: {best_val_loss} ({best_config}) ")

    ### TEST/COMPARISON MODE ###
    elif args.stage == "test":
        # Holds predictions of each model to test
        models_preds = {}

        filename = "last" if args.no_validation else "best-model"

        # Obtain predictions of each trained model
        for model_name in args.model:
            if args.ckpt_path:
                ckpt_path = Path(args.ckpt_path)
                run_dir = ckpt_path.resolve().parents[1]  # .../checkpoints -> run_dir
            else:
                ckpt_best = run_dir / "checkpoints" / "best.ckpt"
                ckpt_last = run_dir / "checkpoints" / "last.ckpt"
                if args.no_validation:
                    ckpt_path = ckpt_last
                else:
                    ckpt_path = ckpt_best if ckpt_best.exists() else ckpt_last

            if not ckpt_path.exists():
                raise FileNotFoundError(f"[TEST] checkpoint not found: {ckpt_path}")

            # print(f"[TEST] Loading checkpoint: {ckpt_path}")
            nusers_for_model = 1 if model_name in ["PRESLEY", "COLLEI"] else None
            model = utils.get_model(model_name, vars(args), nusers_for_model)


            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

            incomp = model.load_state_dict(state_dict, strict=False)
            if incomp.missing_keys or incomp.unexpected_keys:
                # print("Missing keys:", incomp.missing_keys)
                # print("Unexpected keys:", incomp.unexpected_keys)
                raise RuntimeError("Checkpoint incompatible con el modelo actual")

            model.eval()
            test_preds = torch.cat(trainer.predict(model=model, datamodule=dm))
            test_preds = test_preds.detach().cpu().numpy()

            models_preds[model_name] = test_preds

        # Obtain random predictions for baseline comparison
        models_preds["RANDOM"] = (
            torch.mean(torch.rand((len(dm.test_dataset), 10)), dim=1).cpu().numpy()
        )

        cnt = get_centroid_preds(dm)
        models_preds["CNT"] = cnt.cpu().numpy() if torch.is_tensor(cnt) else cnt

        metrics = test_tripadvisor_authorship_task(dm, models_preds, args)

        with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            for model_name, m in metrics.items():
                for k, v in m.items():
                    if v is not None:
                        trainer.logger.experiment.add_scalar(f"test/{model_name}/{k}", float(v), 0)

