import darknet


print(dir(darknet))
# 定义配置文件和权重文件路径
config_path = "50k.cfg"
weights_path = "50k.weights"

# 加载网络
network, class_names, class_colors = darknet.load_network(
    config_path,
    "50k.names",  # 模型类别定义文件
    weights_path,
    batch_size=1
)

# 获取输出层的信息
def print_layer_output_shape(net):
    for i, layer in enumerate(net.layers):
        print(f"Layer {i}: {layer.type}")
        try:
            print(f" - Output Shape: {layer.out_w} x {layer.out_h} x {layer.out_c}")
        except AttributeError:
            print(" - Layer doesn't have output shape attributes")

# 打印每一层的输出形状
print_layer_output_shape(network)
