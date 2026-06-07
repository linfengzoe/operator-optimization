import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def test_setup_declares_all_extensions_in_one_setup_call():
    tree = parse("pytorch/setup.py")

    setup_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(setup_calls) == 1

    setup_source = (ROOT / "pytorch/setup.py").read_text(encoding="utf-8")
    for name in [
        "conv2d_baseline_fp32",
        "conv2d_optim_fp32",
        "conv2d_optim_fp16",
        "conv2d_optim_wmma",
    ]:
        assert name in setup_source


def test_inference_exposes_cli_and_runtime_model_selection():
    tree = parse("inference.py")
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert {"parse_args", "select_resnet18", "main"}.issubset(function_names)

    source = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "if MODEL_DTYPE ==" not in source
    assert "MODEL_DTYPE =" not in source


def test_wmma_backward_fails_loudly_instead_of_returning_zero_gradients():
    source = (ROOT / "cpp/conv2d_optim_wmma.cpp").read_text(encoding="utf-8")

    assert "zeros_like(input)" not in source
    assert "zeros_like(weight)" not in source
    assert "TORCH_CHECK(false" in source
    assert "forward-only" in source


def test_fp16_autograd_saves_tensors_for_backward():
    source = (ROOT / "modules/conv_layer_optim_fp16.py").read_text(encoding="utf-8")

    assert "ctx.save_for_backward" in source
    assert "ctx.saved_tensors" in source
    assert "ctx.saved_variables" not in source
