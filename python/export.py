from train import SimPL, SimpleCNNNet
import torch
import onnx
import onnxruntime as ort
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

DEFAULT_CHECKPOINT = Path(r"C:\Users\Oleksii\Documents\OnlineEducation\ComputerVision\cifar_classify\lightning_logs\cifar10\version_11\checkpoints\epoch=49-step=39100.ckpt")
DEFAULT_ONNX_DEST = Path(Path(__file__).parent.parent, "models", "model.onnx")

def export_to_onnx(checkpoint: Path, output: Path) -> None:
	model_pl = SimPL.load_from_checkpoint(checkpoint, map_location="cpu")
	model = model_pl.model.eval()

	# Export with softmax head
	model.head = torch.nn.Softmax(dim=1)

	dummy_input = torch.rand(1, 3, 32, 32)

	# Export to ONNX
	torch.onnx.export(
		model,
		dummy_input,
		output,
		opset_version=12,
		verbose=True,
		dynamo=False, # New stuff does not work :(
		optimize=True,
		profile=True,
		verify=True,
		input_names=['input'],
		output_names=['output'],
		dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
	)

def raise_if_fail(onnx_model: Path) -> None:
	# Check ONNX model structure
	model = onnx.load(onnx_model)
	onnx.checker.check_model(model, full_check=True)

	# Run inference with ONNX Runtime
	ort_session = ort.InferenceSession(onnx_model)
	dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
	outputs = ort_session.run(["output"], {"input": dummy_input})
	output = outputs[0]

	assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"

def parse_arguments() -> Namespace:
	parser = ArgumentParser(description="Export trained model to ONNX format")
	parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT, help='Path to model checkpoint .ckpt')
	parser.add_argument('--output', type=Path, default=DEFAULT_ONNX_DEST, help='Output ONNX file path')
	return parser.parse_args()

def main() -> None:
	args = parse_arguments()
	export_to_onnx(args.checkpoint, args.output)
	raise_if_fail(args.output)

if __name__ == "__main__":
	main()
