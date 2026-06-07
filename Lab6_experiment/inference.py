import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


DEFAULT_BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def parse_batch_sizes(value):
    try:
        batch_sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc

    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must contain positive integers")
    return batch_sizes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run CIFAR-10 ResNet-18 inference benchmarks.")
    parser.add_argument(
        "--model",
        choices=("baseline", "fp32", "fp16", "wmma"),
        default="fp32",
        help="custom convolution implementation to benchmark",
    )
    parser.add_argument("--model-path", default="./pytorch/model/net_123.pth")
    parser.add_argument("--data-root", default="./pytorch/data")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--baseline-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-times", type=int, default=10)
    parser.add_argument("--accuracy-samples", type=int, default=50)
    parser.add_argument("--accuracy-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true", help="render prediction and throughput plots")
    parser.add_argument("--save-plots-dir", default=None, help="directory for saving plots instead of showing them")
    parser.add_argument("--skip-baseline", action="store_true", help="skip baseline throughput reference")
    parser.add_argument("--no-download", action="store_false", dest="download", help="disable CIFAR-10 download")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.set_defaults(download=True)
    return parser.parse_args(argv)


def select_resnet18(model):
    try:
        if model == "baseline":
            from modules.resnet_18_baseline_fp32 import ResNet18
        elif model == "fp32":
            from modules.resnet_18_optim_fp32 import ResNet18
        elif model == "fp16":
            from modules.resnet_18_optim_fp16 import ResNet18
        elif model == "wmma":
            from modules.resnet_18_optim_wmma import ResNet18
        else:
            raise ValueError(f"invalid model: {model}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Custom convolution extension is missing. Build it first with "
            "`python ./pytorch/setup.py build_ext --inplace` from the Lab6_experiment root."
        ) from exc

    return ResNet18


def model_dtype(model):
    return torch.half if model in {"fp16", "wmma"} else torch.float32


def load_model(model_path, model, device):
    resnet18 = select_resnet18(model)
    net = resnet18()
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net = net.to(device, dtype=model_dtype(model))
    net.eval()
    return net


def build_testset(data_root, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    return torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform,
    )


def denormalize(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
    return tensor.cpu() * std + mean


def evaluate_accuracy(model, dataset, device, model_name, sample_count, batch_size, num_workers, show_progress):
    if sample_count > 0:
        count = min(sample_count, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(count))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    correct = 0
    total = 0
    preview_images = []
    preview_labels = []
    preview_preds = []
    dtype = model_dtype(model_name)
    iterable = tqdm(loader, desc="accuracy", disable=not show_progress)

    with torch.no_grad():
        for images, labels in iterable:
            images = images.to(device=device, dtype=dtype)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            correct += (preds == labels).sum().item()
            total += labels.numel()

            if len(preview_images) < 50:
                remaining = 50 - len(preview_images)
                preview_images.extend(images[:remaining].cpu().float())
                preview_labels.extend(labels[:remaining].cpu())
                preview_preds.extend(preds[:remaining].cpu())

    return correct, total, preview_images, preview_labels, preview_preds


def process_single_run(model, dataloader, device, dtype, show_progress):
    total_images = len(dataloader.dataset)

    with torch.no_grad():
        warmup_tensor = torch.randn(
            dataloader.batch_size,
            3,
            32,
            32,
            dtype=dtype,
            device=device,
        )
        for _ in range(5):
            model(warmup_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=f"reasoning (bs={dataloader.batch_size})",
            ncols=100,
            bar_format="{l_bar}{bar} [{elapsed}<{remaining}]",
            disable=not show_progress,
        )

        for images, _ in progress_bar:
            images = images.to(device=device, dtype=dtype)
            model(images)

            processed = min((progress_bar.n + 1) * dataloader.batch_size, total_images)
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            progress_bar.set_postfix({"speed": f"{processed / elapsed:.1f} img/s"})

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    return total_images / elapsed


def benchmark_model(model, dataset, device, model_name, batch_sizes, num_workers, run_times, show_progress):
    dtype = model_dtype(model_name)
    throughputs = []

    for batch_size in batch_sizes:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        run_values = []
        print(f"\nTesting {model_name} batch_size = {batch_size}")
        for run in range(run_times):
            throughput = process_single_run(model, loader, device, dtype, show_progress)
            run_values.append(throughput)
            print(f"  run {run + 1}/{run_times}: {throughput:.2f} img/s")

        avg_throughput = sum(run_values) / len(run_values)
        throughputs.append(avg_throughput)
        print(f"[{model_name}] Batch Size={batch_size:3d} | Average Throughput: {avg_throughput:.2f} img/s")

    return throughputs


def plot_predictions(images, labels, preds, classes, output_dir):
    if not images:
        return

    plt.figure(figsize=(15, 12))
    for i, (image, label, pred) in enumerate(zip(images, labels, preds), start=1):
        plt.subplot(5, 10, i)
        img = denormalize(image).clamp(0, 1)
        plt.imshow(img.permute(1, 2, 0))

        true_label = classes[int(label)]
        pred_label = classes[int(pred)]
        title_color = "red" if true_label != pred_label else "black"
        plt.title(f"Label: {true_label}\nPred: {pred_label}", color=title_color, fontsize=9)
        plt.axis("off")

    plt.suptitle("Model Prediction Visualization", y=0.99, fontsize=14)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / "predictions.png", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_throughput(batch_sizes, throughputs, baseline_throughput, output_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(
        batch_sizes,
        throughputs,
        marker="o",
        linestyle="-",
        color="#FF6F00",
        linewidth=2,
        markersize=10,
        label="Selected model",
    )

    if baseline_throughput is not None:
        plt.axhline(
            y=baseline_throughput,
            color="#1F77B4",
            linestyle="--",
            linewidth=2,
            label="Baseline (Batch Size=128)",
        )

    if 128 in batch_sizes:
        idx = batch_sizes.index(128)
        plt.scatter([128], [throughputs[idx]], color="red", zorder=5)

    plt.title("Selected Model Throughput vs Baseline", fontsize=14, pad=20)
    plt.xlabel("Batch Size", fontsize=12, labelpad=10)
    plt.ylabel("Throughput (images/sec)", fontsize=12, labelpad=10)
    plt.xticks(batch_sizes, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="upper left")

    for x, y in zip(batch_sizes, throughputs):
        plt.text(x, y + 50, f"{y:.1f}", ha="center", va="bottom", fontsize=10, color="#FF6F00")

    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(output_dir) / "throughput.png", dpi=200)
        plt.close()
    else:
        plt.show()


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("These custom convolution operators require CUDA tensors; CPU fallback is not supported.")
    if args.run_times <= 0:
        raise ValueError("--run-times must be positive")
    if args.baseline_batch_size <= 0:
        raise ValueError("--baseline-batch-size must be positive")
    if args.accuracy_batch_size <= 0:
        raise ValueError("--accuracy-batch-size must be positive")

    dataset = build_testset(args.data_root, download=args.download)
    selected_model = load_model(args.model_path, args.model, device)

    correct, total, preview_images, preview_labels, preview_preds = evaluate_accuracy(
        selected_model,
        dataset,
        device,
        args.model,
        args.accuracy_samples,
        args.accuracy_batch_size,
        args.num_workers,
        show_progress=not args.no_progress,
    )

    accuracy = correct / total * 100 if total else 0.0
    print("\nAccuracy Breakdown:")
    print("-------------------")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    for i, (label, pred) in enumerate(zip(preview_labels, preview_preds), start=1):
        true_label = CLASSES[int(label)]
        pred_label = CLASSES[int(pred)]
        status = "OK" if true_label == pred_label else "MISS"
        print(f"Sample {i:2d}: Pred={pred_label:10s} | True={true_label:10s} | {status}")

    baseline_throughput = None
    if not args.skip_baseline and args.model != "baseline":
        baseline_model = load_model(args.model_path, "baseline", device)
        baseline_loader = DataLoader(
            dataset,
            batch_size=args.baseline_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("\nTesting baseline reference:")
        baseline_throughput = process_single_run(
            baseline_model,
            baseline_loader,
            device,
            torch.float32,
            show_progress=not args.no_progress,
        )
        print(
            f"[baseline] Batch Size={args.baseline_batch_size} | "
            f"Throughput: {baseline_throughput:.2f} img/s"
        )

    throughputs = benchmark_model(
        selected_model,
        dataset,
        device,
        args.model,
        args.batch_sizes,
        args.num_workers,
        args.run_times,
        show_progress=not args.no_progress,
    )

    if args.plot or args.save_plots_dir:
        plot_predictions(preview_images, preview_labels, preview_preds, CLASSES, args.save_plots_dir)
        plot_throughput(args.batch_sizes, throughputs, baseline_throughput, args.save_plots_dir)


if __name__ == "__main__":
    main()
