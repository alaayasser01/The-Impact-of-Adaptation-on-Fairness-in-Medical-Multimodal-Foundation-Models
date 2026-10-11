import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.base import BaseTrainer
from utils.metrics import evaluate_binary, organize_results
from utils.basics import creat_folder
from utils.lr_sched import adjust_learning_rate
from datasets.utils import get_dataset


class CLSTrainer(BaseTrainer):
    def __init__(self, args, model, test_loader, logger, val_loader=None) -> None:
        super().__init__(args, model, logger)
        self.test_loader = test_loader
        if val_loader is not None:
            self.val_loader = val_loader

    def train(self, train_dataloader, val_dataloader=None):
        self.model.train()
        best_val_auc = 0.0
        best_epoch = 0
        best_state = None
        patience_counter = 0
        patience = 5

        while self.epoch < self.total_epochs:
            adjust_learning_rate(self.optimizer, self.epoch + 1, self.args)

            loss = self.train_epoch(train_dataloader)
            self.logger.info("epoch {}: lr {}, loss {}".format(
                self.epoch, self.optimizer.param_groups[0]["lr"], loss))

            # Validation
            if val_dataloader is not None:
                auc = self.evaluate(val_dataloader, return_auc=True)
                self.logger.info(f"epoch {self.epoch}: Val AUC = {auc:.4f}")

                if auc > best_val_auc:
                    best_val_auc = auc
                    best_epoch = self.epoch
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    torch.save(
                        {"model": best_state, "optimizer": self.optimizer.state_dict(), "epoch": self.epoch},
                        os.path.join(self.args.save_folder, "best_ckpt.pth"),
                    )
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(
                        f"Early stopping at epoch {self.epoch}, best val AUC {best_val_auc:.4f} at epoch {best_epoch}"
                    )
                    break

            self.epoch += 1

            # Per-epoch test evaluation (if flag enabled)
            if self.args.save_test_per_epoch and self.test_loader is not None:
                self.args.sensitive_name = "All"
                _, test_loader_all, _ = get_dataset(self.args, split="test")
                self.evaluate(test_loader_all)

                if self.args.dataset == "FairVLMed10k":
                    attrs = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
                elif self.args.dataset == "CXP":
                    attrs = ["Sex", "Age"]
                elif self.args.dataset == "DDI":
                    attrs = ["fitzpatrick"]
                else:
                    attrs = ["Sex"]

                for attr in attrs:
                    self.args.sensitive_name = attr
                    _, test_loader_attr, _ = get_dataset(self.args, split="test")
                    self.evaluate(test_loader_attr)

                self.args.sensitive_name = "All"

        # Load best checkpoint after training
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.logger.info(f"Loaded best checkpoint from epoch {best_epoch} with val AUC {best_val_auc:.4f}")

    def train_epoch(self, train_dataloader):
        loss_epoch = []
        for minibatch in train_dataloader:
            if hasattr(train_dataloader, "class_weights_y"):
                loss_batch = self.update_batch(minibatch, train_dataloader.class_weights_y)
            else:
                loss_batch = self.update_batch(minibatch, None)
            loss_epoch.append(loss_batch.item())
        return np.mean(loss_epoch)

    def update_batch(self, minibatch, class_weights=None):
        x = minibatch["img"].to(self.device)
        y = minibatch["label"].to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(
            logits,
            y.long().squeeze(-1),
            weight=class_weights.to(self.device) if class_weights is not None else None,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss

    def evaluate(self, dataloader, save_path=None, return_auc=False):
        self.model.eval()
        logits_list, prob_list, target_list, sensitive_list = [], [], [], []
        for minibatch in dataloader:
            x = minibatch["img"].to(self.device)
            y = minibatch["label"].to(self.device)
            a = minibatch["sensitive"].to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                if torch.isnan(logits).any():
                    self.logger.warning("NaN detected in logits during evaluation!")
                    logits = torch.zeros_like(logits)
            prob = F.softmax(logits, dim=-1)
            logits_list.append(logits)
            prob_list.append(prob)
            target_list.append(y)
            sensitive_list.append(a)

        logits_arr = torch.concat(logits_list).squeeze().cpu().numpy()
        prob_arr = torch.concat(prob_list).squeeze().cpu().numpy()
        target_arr = torch.concat(target_list).squeeze().cpu().numpy().astype(int)
        sens_arr = torch.concat(sensitive_list).squeeze().cpu().numpy().astype(int)

        overall, subgroup = evaluate_binary(prob_arr[:, 1], target_arr, sens_arr)
        organized = organize_results(overall, subgroup)

        if self.args.early_stopping == 1:
            if len(self.last_five_auc) >= 5:
                self.last_five_auc.pop(0)
            self.last_five_auc.append(overall["auc"])

        self.logger.info("----------------------------------------------")
        self.logger.info("----------------eva epoch {}------------------".format(self.epoch))
        self.logger.info(
            "{}".format(
                ", ".join("{}: {}".format(k, v) for k, v in organized.items()),
            )
        )
        self.logger.info("-----------------meta info-------------------")
        self.logger.info(
            "overall metrics: {}".format(
                ", ".join("{}: {}".format(k, v) for k, v in overall.items()),
            )
        )
        self.logger.info(
            "subgroup metrics: {}".format(
                ", ".join("{}: {}".format(k, v) for k, v in subgroup.items()),
            )
        )
        self.logger.info("----------------------------------------------")

        if save_path:
            creat_folder(save_path)
            with open(os.path.join(save_path, "metrics.pkl"), "wb") as f:
                pickle.dump({"epoch": self.epoch, "overall": overall, "subgroup": subgroup}, f)
            with open(os.path.join(save_path, "predictions.pkl"), "wb") as f:
                pickle.dump({"epoch": self.epoch, "logits": logits_arr, "label": target_arr}, f)

        if return_auc:
            return overall["auc"]