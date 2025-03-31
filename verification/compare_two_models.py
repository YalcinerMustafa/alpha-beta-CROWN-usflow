import onnxruntime as ort
import numpy as np
import argparse


def get_input_shape(session):
    """Get the input shape from the first input of the ONNX model session.
       Replace dynamic dimensions (None) with 1.
    """
    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape
    # Replace any dynamic dimensions with 1
    fixed_shape = [dim if isinstance(dim, int) and dim is not None else 1 for dim in input_shape]
    return input_meta.name, fixed_shape


def compare_outputs(outputs1, outputs2, atol=1e-3):
    """Compare two lists of outputs and print the maximum difference for each output."""
    for idx, (out1, out2) in enumerate(zip(outputs1, outputs2)):
        print(f"Comparing output {idx}:")
        if np.allclose(out1, out2, atol=atol):
            print("  Outputs are similar.")
        else:
            diff = np.abs(out1 - out2)
            max_diff = np.max(diff)
            print(f"  Outputs differ. Maximum absolute difference: {max_diff}")


def main(model1_path, model2_path):
    random = np.random
    for i in range(10):
        # Create ONNX Runtime sessions for both models.
        session1 = ort.InferenceSession(model1_path)
        session2 = ort.InferenceSession(model2_path)


        # Get input name and shape from the first model.
        input_name, input_shape = get_input_shape(session1)
        print(f"Using input '{input_name}' with shape {input_shape}")

        # Generate random input data (using normal distribution).
        random_input = random.randn(*input_shape).astype(np.float32)

        # Run inference on both models.
        outputs1 = session1.run(None, {input_name: random_input})
        outputs2 = session2.run(None, {input_name: random_input})

        # Compare the outputs.
        compare_outputs(outputs1, outputs2)


if __name__ == "__main__":
    model1 = "/home/mustafa/repos/abcrown/alpha-beta-CROWN/models/model_conv.onnx"
    model2 = "/home/mustafa/repos/abcrown/alpha-beta-CROWN/models/model_conv_modified.onnx"
    main(model1, model2)
