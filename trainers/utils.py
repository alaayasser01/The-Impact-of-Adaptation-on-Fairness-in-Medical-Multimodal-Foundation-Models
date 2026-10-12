from trainers import CLSTrainer


def get_trainer(args, model, logger, test_dataloader=None, val_dataloader=None):
    if args.task == "cls":
        if args.method == "erm":
            return CLSTrainer(args, model, test_dataloader, logger, val_dataloader)
        else:
            raise NotImplementedError(f"Method '{args.method}' not implemented.")
    else:
        raise NotImplementedError(f"Task '{args.task}' not supported.")
