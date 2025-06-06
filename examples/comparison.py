import argparse
import csv
import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from data import (
    CIFAR10DataModule,
    HIGGSDataModule,
    MNISTDataModule,
    ShakespeareDataModule,
)
from flax import linen

import pjax
from pjax import nn, optim


# --- Model Definitions ---
class MLP(linen.Module):
    """Flax MLP model with optional skip connections.

    Attributes:
        hidden_features: list of integers specifying the number of units in each hidden layer.
        classes: number of output classes for classification.
        skip: whether to use skip connections by concatenating all hidden layer outputs.
    """

    hidden_features: list[int]
    classes: int = 10
    skip: bool = True

    @linen.compact
    def __call__(self, x):
        # Flatten input for MLP
        x = x.reshape((x.shape[0], -1))
        xs = []
        for features in self.hidden_features:
            x = linen.Dense(features, kernel_init=linen.initializers.he_normal())(x)
            x = linen.relu(x)
            xs.append(x)

        if self.skip:
            x = jnp.concatenate(xs, axis=-1)

        return linen.Dense(self.classes, kernel_init=linen.initializers.he_normal())(x)


class MLP_pjax(nn.Module):
    """PJAX MLP model with optional skip connections.

    Args:
        hidden_features: list of integers specifying hidden layer sizes.
        in_features: number of input features (flattened input size).
        classes: number of output classes for classification.
        skip: whether to concatenate all hidden layer outputs before final layer.
    """

    def __init__(self, hidden_features, in_features, classes, skip=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        last_f = in_features
        for i, f in enumerate(hidden_features):
            setattr(self, f"dense_{i}", nn.Linear(last_f, f))
            setattr(self, f"relu_{i}", nn.ReLU(f))
            last_f = f

        out_features = sum(hidden_features) if skip else hidden_features[-1]
        self.out = nn.Linear(out_features, classes)

    def __call__(self, x):
        # Flatten input for MLP
        x = pjax.reshape(x, (x.shape[0], -1))
        xs = []
        for i in range(len(self.hidden_features)):
            x = getattr(self, f"dense_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            xs.append(x)

        if self.skip:
            x = pjax.concatenate(xs, axis=-1)

        return self.out(x)


class CNN(linen.Module):
    """Flax CNN model with optional skip connections and max pooling.

    Attributes:
        hidden_features: list of integers specifying channel sizes for each conv layer.
        classes: number of output classes for classification.
        skip: whether to concatenate features from all conv layers.
        max_pool: whether to apply max pooling over spatial dimensions.
        stride: convolution stride for all layers.
    """

    hidden_features: list[int]
    classes: int = 10
    skip: bool = True
    max_pool: bool = True
    stride: int = 1

    @linen.compact
    def __call__(self, x):
        xs = []
        for features in self.hidden_features:
            x = linen.Conv(features, (3, 3), (self.stride, self.stride), "SAME")(x)
            x = linen.relu(x)
            if self.max_pool:
                xs.append(jnp.max(x, axis=(1, 2)))
            else:
                xs.append(jnp.reshape(x, (x.shape[0], -1)))

        if self.skip:
            x = jnp.concatenate(xs, axis=-1)
        else:
            x = xs[-1]

        return linen.Dense(self.classes, kernel_init=linen.initializers.he_normal())(x)


class CNN_pjax(nn.Module):
    """PJAX CNN model with optional skip connections and max pooling.

    Args:
        hidden_features: list of integers specifying channel sizes for conv layers.
        in_features: number of input channels.
        size_2d: spatial size of square input images (height = width).
        classes: number of output classes for classification.
        skip: whether to concatenate features from all conv layers.
        max_pool: whether to apply max pooling over spatial dimensions.
        stride: convolution stride for all layers.
    """

    def __init__(self, hidden_features, in_features, size_2d, classes, skip=True, max_pool=True, stride=1):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        self.max_pool = max_pool
        last_f = in_features
        for i, f in enumerate(hidden_features):
            setattr(self, f"conv_{i}", nn.Conv2D(last_f, f, (3, 3), (stride, stride), "SAME"))
            setattr(self, f"relu_{i}", nn.ReLU(f))
            last_f = f

        # calculate output features for the final dense layer
        out_features = 0
        current_h, current_w = size_2d, size_2d
        for f in hidden_features:
            current_h = (current_h + stride - 1) // stride
            current_w = (current_w + stride - 1) // stride
            out_features += f if self.max_pool else current_h * current_w * f
        if not self.skip:
            out_features = hidden_features[-1] if max_pool else current_h * current_w * hidden_features[-1]

        self.out = nn.Linear(out_features, classes)

    def __call__(self, x):
        xs = []
        for i in range(len(self.hidden_features)):
            x = getattr(self, f"conv_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            if self.max_pool:
                xs.append(pjax.max(x, axis=(1, 2)))
            else:
                xs.append(pjax.reshape(x, (x.shape[0], -1)))

        if self.skip:
            x = pjax.concatenate(xs, axis=-1)
        else:
            x = xs[-1]

        return self.out(x)


class RNN(linen.Module):
    """Flax RNN model with optional skip connections.

    Attributes:
        hidden_features: list of integers specifying MLP cell architecture.
        vocab_size: size of the vocabulary for token embeddings.
        skip: whether to use skip connections in the MLP cell.
    """

    hidden_features: list[int]
    vocab_size: int
    skip: bool = True

    @linen.compact
    def __call__(self, x):
        state_size = self.hidden_features[0]
        cell = MLP(self.hidden_features, state_size + self.vocab_size, skip=self.skip)
        x = linen.Embed(self.vocab_size, state_size)(x)
        h = jnp.zeros((x.shape[0], state_size))
        out = []
        for t in range(x.shape[1]):
            cell_out = cell(jnp.concatenate((x[:, t], h), axis=-1))
            h = cell_out[:, :state_size]
            h = jax.nn.relu(h)
            out.append(cell_out[:, state_size:])
        return jnp.stack(out, axis=1)


class RNN_pjax(nn.Module):
    """PJAX RNN model with optional skip connections.

    Args:
        hidden_features: list of integers specifying MLP cell architecture.
        vocab_size: size of the vocabulary for token embeddings.
        skip: whether to use skip connections in the MLP cell.
    """

    def __init__(self, hidden_features, vocab_size, skip=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        self.vocab_size = vocab_size
        self.state_size = hidden_features[0]
        self.cell = MLP_pjax(hidden_features, 2 * self.state_size, self.state_size + vocab_size, skip=skip)
        self.embedding = nn.Embedding(vocab_size, self.state_size)

    def __call__(self, x):
        h = jnp.zeros((x.shape[0], self.state_size))
        out = []
        for t in range(x.shape[1]):
            token = pjax.array(x[:, t], name=f"token_{t}")
            emb = self.embedding(token)
            cell_out = self.cell(pjax.concatenate((emb, h), axis=-1))
            h, logits = cell_out[:, : self.state_size], cell_out[:, self.state_size :]
            h = pjax.relu(h)  # required to make graph bipartite
            out.append(logits)
        return pjax.stack(out, axis=1)


# --- Benchmark Function ---
def benchmark(model, optimizer, dataset, seed, margin_loss, patience=10, eval_every=100, max_steps=0):
    """Benchmark a model-optimizer combination on a dataset.

    Args:
        model: either a Flax or PJAX model instance to train.
        optimizer: either an Optax optimizer (for Flax) or PJAX optimizer instance.
        dataset: data module providing train/validation data loaders.
        seed: random seed for reproducible initialization and training.
        margin_loss: whether to use margin loss (PJAX) or cross-entropy loss (Flax).
        patience: number of evaluations without improvement before early stopping.
        eval_every: number of training steps between validation evaluations.
        max_steps: maximum number of training steps (0 for unlimited).

    Returns:
        dict: comprehensive results including final validation accuracy, training time,
            convergence step, validation history, and training statistics.
    """

    key = jax.random.key(seed)

    # --- Data Iterators/Loaders ---
    # Use the dataloaders provided by the DataModule
    train_iter_split = dataset.train_iterator()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()

    # --- Initialization ---
    if isinstance(model, linen.Module):
        # Get an initialization batch from the validation loader (safer than train_iter which might be infinite)
        init_x, _ = next(iter(val_dataloader))
        params = model.init(key, init_x)
        opt_state = optimizer.init(params)

        def step_fn_linen(params, opt_state, x, y):
            def loss_fn(params):
                logits = model.apply(params, x)
                return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

            _, grad = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grad, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state

        step_fn = step_fn_linen

    else:  # pjax model
        params = model.init(key)
        opt_state = None  # pjax optimizers are stateless in this interface

        def step_fn_pjax(params, opt_state, x, y):
            def apply_fn(params):
                pred = model.apply(params, x)
                num_classes = pred.shape[-1]
                y_one_hot = jax.nn.one_hot(y, num_classes)
                if margin_loss:
                    return pjax.margin_loss(pred, y_one_hot)
                return pjax.cross_entropy(pred, y_one_hot)

            new_params = optimizer.update(apply_fn, params)[0]
            return new_params, opt_state

        step_fn = step_fn_pjax

    step_fn = jax.jit(step_fn)

    @jax.jit
    def eval_fn(params, x, y):
        pred = model.apply(params, x)  # x is already preprocessed
        if not isinstance(pred, jnp.ndarray):
            pred = pred.value  # hack to get the value from pjax
        return jnp.mean(jnp.argmax(pred, axis=-1) == y)

    # --- Training Loop ---
    steps_per_update = optimizer.steps_per_update if hasattr(optimizer, "steps_per_update") else 1

    best_val_acc = -1.0
    best_params = None
    best_step = 0
    eval_cycles_since_last_improvement = 0
    val_acc_history = []  # Store (step, val_acc, cumulative_train_time)
    cumulative_train_time = 0.0
    step = 0

    print(
        f"Starting training on {jax.devices()[0].platform}... Evaluating every {eval_every} steps. Patience: {patience} evaluations."
    )
    # Use tqdm for progress, but loop indefinitely until break
    with tqdm.tqdm(initial=step, unit="step") as pbar:
        while True:
            # --- Evaluation Phase ---
            if step % eval_every == 0:
                # Evaluate on the validation set
                val_accs = [eval_fn(params, jnp.array(x), jnp.array(y)) for x, y in val_dataloader]
                current_val_acc = jnp.mean(jnp.array(val_accs))
                val_acc_history.append((step, float(current_val_acc), cumulative_train_time))
                pbar.set_postfix(val_acc=f"{current_val_acc:.4f}", best_val_acc=f"{best_val_acc:.4f}")

                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc
                    best_params = params  # Store the best parameters
                    best_step = step
                    eval_cycles_since_last_improvement = 0  # Reset counter
                    print(f"\nNew best validation accuracy: {best_val_acc:.4f} at step {step}")
                else:
                    eval_cycles_since_last_improvement += 1  # Increment evaluation cycle counter

                if eval_cycles_since_last_improvement >= patience:
                    print(f"\nEarly stopping triggered at step {step}. No improvement for {patience} evaluations.")
                    break

            # --- Training Phase ---
            # Perform steps within one update cycle for pjax optimizers
            inner_step_start_time = time.perf_counter()
            # Get next batch from the training iterator
            x, y = next(train_iter_split)
            x, y = jnp.array(x), jnp.array(y)
            # Use a unique key for each training step if needed, but often not necessary
            # step_key, train_key = jax.random.split(train_key)
            params, opt_state = step_fn(params, opt_state, x, y)
            inner_step_end_time = time.perf_counter()
            if step > 0:
                # we don't want to count the first step due to the jit compilation time
                cumulative_train_time += inner_step_end_time - inner_step_start_time

            step += steps_per_update
            pbar.update(steps_per_update)

            # Safety break for runaway loops (optional)
            if max_steps and step >= max_steps:
                print(f"\nWarning: Reached safety step limit ({max_steps}).")
                break

    # --- Final Evaluation on Test Set ---
    if best_params is None:  # Handle case where no improvement happened before patience ran out
        print("Warning: No improvement observed in validation accuracy. Using last parameters for testing.")
        best_params = params
        best_step = step  # Or 0 if preferred

    print(f"Evaluating best model (from step {best_step}) on the test set...")
    # Use the test_dataloader for final evaluation
    test_accs = [eval_fn(best_params, jnp.array(x), jnp.array(y)) for x, y in test_dataloader]
    final_test_acc = jnp.mean(jnp.array(test_accs))
    print(f"Final test accuracy: {final_test_acc:.4f}")

    return float(final_test_acc), float(best_val_acc), best_step, cumulative_train_time, val_acc_history, step


# --- CSV Logging ---
def log_results(filename, args, metrics):
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as csvfile:
        fieldnames = [
            "timestamp",
            "dataset",
            "model_type",
            "optimizer",
            "hidden_features",
            "skip",
            "batch_size",
            "learning_rate",
            "steps_per_update",
            "dm_beta",
            "margin_loss",
            "patience",
            "eval_every",
            "max_steps",
            "num_runs",
            "seq_len",
            "test_acc_mean",
            "test_acc_std",
            "best_step_mean",
            "best_step_std",
            "total_time_mean",
            "total_time_std",
            "steps_to_conv_mean",
            "steps_to_conv_std",
            "time_to_conv_mean",
            "time_to_conv_std",
            "step_time_ms_mean",
            "step_time_ms_std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": timestamp,
            "dataset": args.dataset,
            "model_type": args.model_type,
            "optimizer": args.optimizer,
            "hidden_features": str(args.hidden_features),
            "skip": args.skip,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate if args.optimizer in ["sgd", "adam"] else "N/A",
            "steps_per_update": 1 if args.optimizer in ["sgd", "adam"] else args.steps_per_update,
            "dm_beta": args.dm_beta if args.optimizer == "dm" else "N/A",
            "margin_loss": args.margin_loss if args.optimizer not in ["sgd", "adam"] else "N/A",
            "patience": args.patience,
            "eval_every": args.eval_every,
            "max_steps": args.max_steps,
            "num_runs": args.num_runs,
            "seq_len": args.seq_len if args.model_type == "rnn" else "N/A",  # Added seq_len value
            "test_acc_mean": f"{metrics['test_acc_mean']:.6f}",
            "test_acc_std": f"{metrics['test_acc_std']:.6f}",
            "best_step_mean": f"{metrics['best_step_mean']:.2f}",
            "best_step_std": f"{metrics['best_step_std']:.2f}",
            "total_time_mean": f"{metrics['total_time_mean']:.4f}",
            "total_time_std": f"{metrics['total_time_std']:.4f}",
            "steps_to_conv_mean": f"{metrics['steps_to_conv_mean']:.2f}",
            "steps_to_conv_std": f"{metrics['steps_to_conv_std']:.2f}",
            "time_to_conv_mean": f"{metrics['time_to_conv_mean']:.4f}",
            "time_to_conv_std": f"{metrics['time_to_conv_std']:.4f}",
            "step_time_ms_mean": f"{metrics['step_time_ms_mean']:.4f}",
            "step_time_ms_std": f"{metrics['step_time_ms_std']:.4f}",
        }
        writer.writerow(row)
    print(f"Aggregated results logged to {filename}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark comparison for different models and optimizers.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "CIFAR10", "HIGGS", "shakespeare"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--model_type", type=str, default="mlp", choices=["mlp", "cnn", "rnn"], help="Type of model architecture."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="dr",
        choices=["sgd", "adam", "ap", "dr", "dm", "ar", "cp"],
        help="Optimizer to use.",
    )
    parser.add_argument("--hidden_features", type=int, nargs="+", default=[32, 32], help="List of hidden layer sizes.")
    parser.add_argument(
        "--skip", action="store_true", default=False, help="Whether to skip all hidden layers to the output layer."
    )
    # Removed num_steps
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate model on validation set every N steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for SGD/Adam.")
    parser.add_argument(
        "--steps_per_update", type=int, default=50, help="Steps per update for pjax optimizers (DR, AP, CP)."
    )
    parser.add_argument("--dm_beta", type=float, default=0.5, help="Beta parameter for Difference Map optimizer.")
    parser.add_argument(
        "--margin_loss",
        action="store_true",
        default=False,
        help="Use threshold margin loss instead of cross-entropy for pjax optimizers.",
    )
    parser.add_argument("--log_file", type=str, default="comparison_results.csv", help="CSV file to log results.")
    # Removed overfit_batches
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of evaluation cycles without validation improvement to trigger early stopping.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=5, help="Number of runs with different seeds for statistical analysis."
    )
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Maximum number of training steps before stopping (0 for no limit)."
    )
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for RNN models (shakespeare dataset).")

    args = parser.parse_args()

    # Pre-determine dataset characteristics needed for model init
    input_features = None
    size_2d = None
    num_classes = None
    if args.dataset == "MNIST":
        input_channels = 1
        img_size = 28
        num_classes = 10
        if args.model_type == "mlp":
            input_features = img_size * img_size * input_channels
        elif args.model_type == "cnn":
            input_features = input_channels
            size_2d = img_size
    elif args.dataset == "CIFAR10":
        input_channels = 3
        img_size = 32
        num_classes = 10
        if args.model_type == "mlp":
            input_features = img_size * img_size * input_channels
        elif args.model_type == "cnn":
            input_features = input_channels
            size_2d = img_size
    elif args.dataset == "HIGGS":
        input_features = 28  # HIGGS dataset has 28 features
        num_classes = 2
    elif args.dataset == "shakespeare":
        num_classes = 65  # number of unique characters in the dataset
        if args.model_type != "rnn":
            raise ValueError("Shakespeare dataset only supports RNN model type.")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.model_type not in ["mlp", "cnn", "rnn"]:
        raise ValueError(f"Unknown model type: {args.model_type}")
    if args.model_type == "rnn" and args.dataset != "shakespeare":
        raise ValueError("RNN model type is only supported for the shakespeare dataset.")

    # --- Run Benchmark Multiple Times ---
    test_accuracies = []
    best_steps = []
    total_times = []
    steps_to_convergence_list = []
    times_to_convergence_list = []
    all_step_times = []

    print(f"\nRunning benchmark {args.num_runs} times...")
    for i in range(args.num_runs):
        seed = i
        print(f"\n--- Run {i+1}/{args.num_runs} ---")

        # --- Data Loading (Inside the loop with unique seed) ---
        print(f"Loading dataset: {args.dataset} for run {i+1}")
        if args.dataset == "MNIST":
            data = MNISTDataModule(batch_size=args.batch_size, preload=True, seed=seed)
        elif args.dataset == "CIFAR10":
            data = CIFAR10DataModule(batch_size=args.batch_size, preload=True, seed=seed)
        elif args.dataset == "HIGGS":
            data = HIGGSDataModule(batch_size=args.batch_size, preload=False, seed=seed)
        elif args.dataset == "shakespeare":
            data = ShakespeareDataModule(batch_size=args.batch_size, seq_len=args.seq_len, preload=False, seed=seed)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # --- Model Selection (recreate model for each run) ---
        print(
            f"Creating model: {args.model_type.upper()} with hidden features {args.hidden_features}, skip: {args.skip}"
        )
        use_linen = args.optimizer in ["sgd", "adam"]
        if args.model_type == "mlp":
            if use_linen:
                model = MLP(hidden_features=args.hidden_features, classes=num_classes, skip=args.skip)
            else:
                model = MLP_pjax(
                    hidden_features=args.hidden_features,
                    in_features=input_features,
                    classes=num_classes,
                    skip=args.skip,
                )
        elif args.model_type == "cnn":
            if use_linen:
                model = CNN(hidden_features=args.hidden_features, classes=num_classes, skip=args.skip)
            else:
                model = CNN_pjax(
                    hidden_features=args.hidden_features,
                    in_features=input_features,
                    size_2d=size_2d,
                    classes=num_classes,
                    skip=args.skip,
                )
        elif args.model_type == "rnn":
            if use_linen:
                model = RNN(hidden_features=args.hidden_features, vocab_size=num_classes, skip=args.skip)
            else:
                model = RNN_pjax(
                    hidden_features=args.hidden_features,
                    vocab_size=num_classes,
                    skip=args.skip,
                )

        # --- Optimizer Selection ---
        # Recreating optimizer is less critical unless it has state, but good practice
        print(f"Using optimizer: {args.optimizer}")
        if args.optimizer == "sgd":
            optimizer = optax.sgd(args.learning_rate)
        elif args.optimizer == "adam":
            optimizer = optax.adam(args.learning_rate)
        elif args.optimizer == "ap":
            optimizer = optim.AlternatingProjections(steps_per_update=args.steps_per_update)
        elif args.optimizer == "dr":
            optimizer = optim.DouglasRachford(steps_per_update=args.steps_per_update)
        elif args.optimizer == "dm":
            optimizer = optim.DifferenceMap(beta=args.dm_beta)
        elif args.optimizer == "ar":
            optimizer = optim.AlternatingReflections(steps_per_update=args.steps_per_update)
        elif args.optimizer == "cp":
            optimizer = optim.CyclicProjections(steps_per_update=args.steps_per_update)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        # --- Run Benchmark for this seed ---
        test_acc, best_val_acc, best_step, total_time, val_acc_history, step = benchmark(
            model,
            optimizer,
            data,
            seed=seed,
            margin_loss=args.margin_loss,
            patience=args.patience,
            eval_every=args.eval_every,
            max_steps=args.max_steps,
        )

        # --- Calculate Convergence Metrics ---
        steps_to_convergence = np.nan
        time_to_convergence = np.nan
        if best_val_acc > 0:  # Avoid division by zero or issues if no improvement
            target_acc = 0.99 * best_val_acc  # convergence threshold: 99% of best validation accuracy
            for step_hist, acc_hist, time_hist in val_acc_history:
                if acc_hist >= target_acc:
                    steps_to_convergence = step_hist
                    time_to_convergence = time_hist
                    break  # Found the first time it reached convergence target

        # --- Store Results ---
        test_accuracies.append(test_acc)
        best_steps.append(best_step)
        total_times.append(total_time)
        steps_to_convergence_list.append(steps_to_convergence)
        times_to_convergence_list.append(time_to_convergence)
        all_step_times.append(total_time / step)

        print(f"Run {i+1} finished. Test Acc: {test_acc:.4f}, Best Step: {best_step}, Total Time: {total_time:.2f}s")
        print(
            f"Steps to Convergence (99% of best val): {steps_to_convergence}, Time to Convergence: {time_to_convergence:.2f}s"
        )

    # --- Aggregate Results ---
    metrics = {
        "test_acc_mean": np.mean(test_accuracies),
        "test_acc_std": np.std(test_accuracies),
        "best_step_mean": np.mean(best_steps),
        "best_step_std": np.std(best_steps),
        "total_time_mean": np.mean(total_times),
        "total_time_std": np.std(total_times),
        "step_time_ms_mean": np.mean(all_step_times) * 1000,
        "step_time_ms_std": np.std(all_step_times) * 1000,
        # Use nanmean/nanstd for convergence metrics as they might be NaN if convergence wasn't reached
        "steps_to_conv_mean": np.nanmean(steps_to_convergence_list),
        "steps_to_conv_std": np.nanstd(steps_to_convergence_list),
        "time_to_conv_mean": np.nanmean(times_to_convergence_list),
        "time_to_conv_std": np.nanstd(times_to_convergence_list),
    }

    print("\n--- Aggregated Results ---")
    print(f"Test Accuracy: {metrics['test_acc_mean']:.4f} +/- {metrics['test_acc_std']:.4f}")
    print(f"Best Step: {metrics['best_step_mean']:.2f} +/- {metrics['best_step_std']:.2f}")
    print(f"Total Training Time: {metrics['total_time_mean']:.2f}s +/- {metrics['total_time_std']:.2f}s")
    print(f"Steps to Convergence: {metrics['steps_to_conv_mean']:.2f} +/- {metrics['steps_to_conv_std']:.2f}")
    print(f"Time to Convergence: {metrics['time_to_conv_mean']:.2f}s +/- {metrics['time_to_conv_std']:.2f}s")
    print(f"Time per Step: {metrics['step_time_ms_mean']:.2f}ms +/- {metrics['step_time_ms_std']:.2f}ms")

    # --- Log Results ---
    log_results(args.log_file, args, metrics)

    print("\nScript finished.")
