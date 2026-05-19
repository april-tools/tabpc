from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.flop_counter import FlopCounterMode
from codecarbon import EmissionsTracker

from src.util import (Dequantizer, NoiseCurriculum, Timer, input_listener,
                      set_seeds)

from .optimization import Optimization


def log_statistic(*, name: str, value) -> None:
    print(f"{name}: {value:4g}")


def loss_plot(
    *,
    train_losses,
    validation_losses=None,
    epoch: int = 0,
    freq: int = 1,
    loss_plot_name="losses.png",
    path: str | None = None,
) -> None:
    file = f"{path}/{loss_plot_name}" if path else loss_plot_name
    if epoch % freq == 0:
        plt.close("all")
        plt.rcdefaults()
        plt.plot(
            np.arange(1, len(train_losses) + 1), train_losses, alpha=0.8, label="train"
        )
        if validation_losses is not None:
            plt.plot(
                np.arange(1, len(validation_losses) + 1),
                validation_losses,
                alpha=0.8,
                label="validation",
            )
        plt.legend()
        plt.savefig(file)
        plt.close("all")


def number_of_parameters(nn: torch.nn.Module) -> int:
    return sum(p.numel() for p in nn.parameters())


def print_number_of_parameters(nn: torch.nn.Module) -> None:
    print(
        "\033[1;34m" + f"Number of parameters: {number_of_parameters(nn)}" + "\x1b[0m"
    )


def get_flops(model, inp, loss_function=None, with_backward=False, grad_accumulation_steps=1):
    istrain = model.training
    model.eval()

    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        if loss_function is not None:
            if with_backward:
                loss = loss_function(X=inp)  # type: ignore
                (loss / grad_accumulation_steps).backward()
            else:
                loss = loss_function(X=inp)  # type: ignore
        else:
            loss = None
            if with_backward:
                model(inp).sum().backward()
            else:
                model(inp)
    total_flops = flop_counter.get_total_flops()
    if istrain:
        model.train()

    return total_flops, loss


def reasonable_improvement(
    old_loss: float | None, new_loss: float, threshold: float = 1e-4
) -> bool:
    if old_loss is None or old_loss == float("inf"):
        return True
    return (old_loss - new_loss) / abs(old_loss) > threshold


def nn_training(
    *,
    train_set: Tensor,
    validation_set: Tensor | None = None,
    optimization: Optimization,
    loss_function: Callable,
    nn: torch.nn.Module,
    device: str | None = None,
    wandb_run=None,
    seed: int | None = None,
    emissions_tracker: EmissionsTracker | None = None,
) -> tuple[list[float], list[float], list[float], int, int, EmissionsTracker | None]:
    tensor_train_set = TensorDataset(train_set.to(device))

    nn.to(train_set.device)
    if optimization.compile_nn:
        nn = torch.compile(nn)

    dequantizer = None
    if optimization.dequantization_noise:
        dequantizer = Dequantizer(train_data=train_set)

    noise_curriculum = None
    if optimization.noise_curriculum:
        noise_curriculum = NoiseCurriculum(
            train_data=train_set, max_noise=1.0, t_max=int(optimization.epochs * 0.75)
        )

    if seed is not None:
        set_seeds(seed)

    train_loader = DataLoader(
        dataset=tensor_train_set,
        batch_size=optimization.batch_size,
        shuffle=True,
    )
    if validation_set is not None:
        tensor_validation_set = TensorDataset(validation_set.to(device))
        validation_loader = DataLoader(
            dataset=tensor_validation_set,
            batch_size=optimization.batch_size,
            shuffle=False,
        )

    nn.train(True)

    optimizer = optimization.build_optimizer(nn.parameters())

    if optimization.lr_scheduler_class is not None:
        lr_scheduler = optimization.build_lr_scheduler(optimizer)  # Example scheduler

    train_epochs_losses = []
    validation_epochs_losses = []
    best_validation_loss = torch.inf
    best_model = None
    epochs_since_best = 0
    n_batches = len(train_loader)

    input_happened = input_listener() if optimization.interactive else False

    epochs_elapsed_times = []

    torch.cuda.memory.reset_peak_memory_stats()

    trailing_nans = 0

    if emissions_tracker is not None:
        emissions_tracker.start()

    for epoch in range(optimization.epochs):
        with Timer() as epoch_timer:
            if input_happened:
                break
            log_statistic(name="epoch", value=epoch + 1)
            print(f'learning rate: {optimizer.param_groups[0]["lr"]:.2e}')
            losses = np.zeros(n_batches)

            pbar = tqdm.tqdm(total=n_batches)
            optimizer.zero_grad()

            for i, X_batch in enumerate(train_loader):
                if input_happened:
                    break

                if noise_curriculum:
                    X_batch[0] = noise_curriculum.add_noise(X_batch[0], t=epoch)

                if dequantizer is not None:
                    X_inp = dequantizer(X_batch[0])
                else:
                    X_inp = X_batch[0]

                if i == 0:
                    train_flops, loss = get_flops(nn, X_inp, loss_function, with_backward=True, grad_accumulation_steps=optimization.grad_accumulation_steps)
                    log_statistic(name="Training FLOPs", value=train_flops)
                else:
                    loss = loss_function(X=X_inp)
                    (loss / optimization.grad_accumulation_steps).backward()

                if optimization.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=optimization.gradient_clipping)  # type: ignore

                if ((i + 1) % optimization.grad_accumulation_steps == 0) or (
                    i == n_batches - 1
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                losses[i] = loss.detach()
                # pbar.set_postfix(loss=f"{losses[i]:.2f}")
                pbar.update(1)
            pbar.close()

            train_epochs_losses.append(np.sum(losses) / len(train_set))
            log_statistic(name="train loss mean", value=train_epochs_losses[-1])

            if validation_set is not None:
                with torch.no_grad():
                    validation_losses = np.zeros(len(validation_loader))
                    for i, X_batch in enumerate(validation_loader):
                        validation_losses[i] = (
                            loss_function(X=X_batch[0]).sum().detach()
                        )
                validation_epochs_losses.append(
                    np.sum(validation_losses) / len(validation_set)
                )
                log_statistic(
                    name="validation loss", value=validation_epochs_losses[-1]
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "train_loss": train_epochs_losses[-1],
                            "validation_loss": validation_epochs_losses[-1],
                        }
                    )
                if (
                    optimization.restore_best_validation_model
                    and reasonable_improvement(
                        old_loss=best_validation_loss,
                        new_loss=validation_epochs_losses[-1],
                    )
                ):
                    best_validation_loss = validation_epochs_losses[-1]
                    best_model = nn.state_dict()
                    epochs_since_best = 0
                epochs_since_best += 1

            val = validation_epochs_losses[-1]
            if not np.isfinite(val):
                trailing_nans += 1
                print(
                    f"Validation loss is NaN or infinite! ({trailing_nans} times in a row)"
                )
            else:
                trailing_nans = 0

            if optimization.lr_scheduler_class is not None:
                lr_scheduler.step(validation_epochs_losses[-1])

            print()

            loss_plot(
                train_losses=train_epochs_losses,
                validation_losses=validation_epochs_losses,
                epoch=epoch,
                freq=1 + optimization.epochs // 10,
            )

            if epoch == 0:
                print_number_of_parameters(nn)

            if (
                optimization.patience is not None
                and epochs_since_best > optimization.patience
            ):
                print(
                    f"Early stopping (validation loss not improving since last {epochs_since_best} epochs)"
                )
                break
        epochs_elapsed_times.append(epoch_timer.elapsed)

        if (
            optimization.patience is not None
            and trailing_nans >= optimization.patience
        ):
            print(
                f"Early stopping (validation loss is NaN or infinite for {trailing_nans} epochs)"
            )
            break

    max_memory = torch.cuda.memory.max_memory_allocated(nn.device)

    if emissions_tracker is not None:
        emissions_tracker.stop()

    log_statistic(name="Max memory allocated", value=max_memory)

    loss_plot(
        train_losses=train_epochs_losses,
        validation_losses=validation_epochs_losses,
        epoch=epoch,
        freq=1,
    )

    if optimization.restore_best_validation_model and best_model is not None:
        print("restoring best model...")
        nn.load_state_dict(best_model)

    nn.eval()
    return train_epochs_losses, validation_epochs_losses, epochs_elapsed_times, train_flops, max_memory, emissions_tracker
