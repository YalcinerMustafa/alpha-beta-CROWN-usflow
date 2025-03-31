import onnx
from onnx import helper, numpy_helper, AttributeProto, TensorProto, GraphProto
import numpy as np

# Load the ONNX model
model = onnx.load("/home/mustafa/repos/abcrown/alpha-beta-CROWN/models/model_conv.onnx")
graph = model.graph

# A list to keep track of new nodes to add.
new_nodes = []

# New list of nodes after modification.
updated_nodes = []

# We'll collect new initializers here.
new_initializers = []
# Iterate over the nodes in the graph.
for node in graph.node:
    # Check if this node is a ConvTranspose and has the attribute output_padding.
    if node.op_type == "ConvTranspose":
        # Find attribute named "output_padding"
        output_padding_attr = None
        for attr in node.attribute:
            if attr.name == "output_padding":
                output_padding_attr = attr
                break

        if output_padding_attr is not None:
            # Convert output_padding attribute to a list (assume it's an ints list).
            # Here, we assume output_padding has two values [pad_H, pad_W].
            pad_vals = list(output_padding_attr.ints)
            if pad_vals != [1, 1]:
                print(f"Skipping node {node.name}: output_padding is not [1,1].")
                updated_nodes.append(node)
                continue

            # Remove the output_padding attribute from the node.
            node.attribute.remove(output_padding_attr)

            # Generate a new intermediate output name.
            conv_out_name = node.output[0]
            pad_out_name = conv_out_name + "_padded"

            # Change the ConvTranspose node's output to this intermediate name.
            node.output[0] = conv_out_name + "_unpadded"

            # Create a Pad node that pads only the spatial dimensions (assuming input shape is [N,C,H,W]).
            # The 'pads' attribute is a list: [pad_N_begin, pad_C_begin, pad_H_begin, pad_W_begin,
            # pad_N_end, pad_C_end, pad_H_end, pad_W_end]
            pads = np.array([0, 0, 0, 0, 0, 0, pad_vals[0], pad_vals[1]], dtype=np.int64)
            # Define a unique name for the pads constant.
            pads_const_name = node.name + "_pads_const"
            pads_initializer = numpy_helper.from_array(pads, name=pads_const_name)
            new_initializers.append(pads_initializer)
            pad_node = helper.make_node(
                "Pad",
                inputs=[node.output[0],pads_const_name],
                outputs=[pad_out_name],
                name=node.name + "_pad",
                mode="constant",  # Using constant padding
            )

            # Append the pad node to the new nodes list.
            new_nodes.append(pad_node)

            # For all subsequent nodes that use the ConvTranspose output, replace input name.
            # We do this after iterating over all nodes.
        updated_nodes.append(node)
    else:
        updated_nodes.append(node)

graph.initializer.extend(new_initializers)

# Replace the graph nodes with our updated list.
# Now we add the new Pad nodes to the list.
graph.ClearField("node")
graph.node.extend(updated_nodes + new_nodes)

# Now, fix the inputs of nodes that used the original ConvTranspose output.
# For each node, if an input equals the original conv output name (the one before padding),
# replace it with the pad node output name.
orig_name_suffix = "_unpadded"
for node in graph.node:
    for idx, input_name in enumerate(node.input):
        if input_name == "/conditioner/nn/nn.11/ConvTranspose_output_0":
            # Replace it with the padded name.
            node.input[idx] = input_name.replace(orig_name_suffix, "") + "_padded"

# Save the modified model
onnx.save(model, "/home/mustafa/repos/abcrown/alpha-beta-CROWN/models/model_conv_modified.onnx")
print("Model modification complete. Saved as 'model_modified.onnx'.")
