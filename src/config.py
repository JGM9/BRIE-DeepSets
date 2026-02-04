from argparse import ArgumentParser, BooleanOptionalAction

def read_args():
    parser = ArgumentParser()

    # Required args #
    parser.add_argument("--city", type=str)
    parser.add_argument("--stage", type=str)
    parser.add_argument("--model", type=str, nargs="+")

    parser.add_argument("--batch_size", type=int, default=2**15)
    parser.add_argument("--max_epochs", type=int, default=100)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("-d", type=int, default=256)

    # MPORTANTES: defaults explícitos
    parser.add_argument("--no_validation", action=BooleanOptionalAction, default=False)
    parser.add_argument("--log_to_csv", action=BooleanOptionalAction, default=False)
    parser.add_argument("--early_stopping", action=BooleanOptionalAction, default=False)
    parser.add_argument("--load_preds", action=BooleanOptionalAction, default=False)
    parser.add_argument("--use_train_val", action=BooleanOptionalAction, default=False)

    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--tau", type=float, default=1)

    parser.add_argument("--logdir_name", type=str)

    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--debug_sanity", action=BooleanOptionalAction, default=False)
    parser.add_argument("--debug_sanity_freq", type=int, default=200)
    parser.add_argument("--debug_overfit", action=BooleanOptionalAction, default=False)

    parser.add_argument("--limit_users", type=int)
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--limit_test_batches", type=int)

    parser.add_argument("--fast_dev_run", action=BooleanOptionalAction, default=False)

    parser.add_argument("--ds_no_rho", action=BooleanOptionalAction, default=False)

    parser.add_argument("--save_predictions", action=BooleanOptionalAction, default=False)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_user_images", type=int, default=20)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--ckpt_path", type=str, default="")

    return parser.parse_args()
