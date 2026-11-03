import copy

from moviad.trainers.trainer import Trainer
from moviad.utilities.evaluation.evaluator import Evaluator

# training with early stopping

class EarlyStoppingTrainer(Trainer):
    """MoViAD Trainer that evaluates on the validation set after every epoch,
    keeps the best model and stops when the monitored metric has not improved for `patience` epochs.
    At the end the model holds the weights of the best epoch; the last weights are kept in `last_state`."""

    def __init__(self, *args, patience=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.monitor = self.metrics[0].name
        self.best_score, self.best_epoch = -1.0, -1
        self.history, self.last_state = [], None

    def train(self):
        self.train_args.init_train(self.model)
        best_state, epochs_without_improvement = None, 0

        for epoch in range(self.train_args.epochs):
            self.model.train()
            print(f"EPOCH: {epoch}")
            train_loss = self.model.train_epoch(epoch, self.train_dataloader, self.train_args)
            # model outputs are probabilities, so no extra normalisation
            results = Evaluator.evaluate(self.model, self.eval_dataloader, self.metrics, self.device,
                                         postprocess=lambda x: x)
            score = float(results[self.monitor])
            self.history.append({"epoch": epoch, "train_loss": train_loss,
                                 **{name: float(value) for name, value in results.items()}})
            print(f"train loss {train_loss:.4f}, validation {self.monitor} {score:.4f}")

            if score > self.best_score:
                self.best_score, self.best_epoch = score, epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch}, best epoch {self.best_epoch}")
                    break

        self.last_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
