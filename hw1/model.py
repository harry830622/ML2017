class Model:
    def __init__(self, bias, feature_config):
        self.bias = bias
        self.feature_config = feature_config

    def calculate_y(self, feature_values):
        y = self.bias
        for k, v in self.feature_config.items():
            index_name = k.split("-")[0]
            hour = int(k.split("-")[1].split("^")[0])
            power = int(k.split("^")[1])
            weight = v
            y += weight * (feature_values[index_name][hour - 1] ** power)
        return y

    def calculate_loss_root(self, feature_values, real_y):
        return real_y - self.calculate_y(feature_values)

    def calculate_bias_gradient(self, feature_values, real_y):
        return 2 * self.calculate_loss_root(feature_values, real_y) * (-1)

    def calculate_feature_gradient(self, feature_values, real_y, feature_name):
        index_name = feature_name.split("-")[0]
        hour = int(feature_name.split("-")[1].split("^")[0])
        power = int(feature_name.split("^")[1])
        feature_value = feature_values[index_name][hour - 1]
        return 2 * self.calculate_loss_root(feature_values, real_y) * (-1) * (
                feature_value ** power)
