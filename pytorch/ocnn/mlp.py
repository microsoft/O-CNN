import torch
import torch.nn


class FC(torch.nn.Module):
  def __init__(self, in_features, out_features, act=torch.nn.ReLU(inplace=True)):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features, bias=True)
    self.act = act

  def forward(self, input):
    output = self.linear(input)
    output = self.act(output)
    return output


class FcBn(torch.nn.Module):
  def __init__(self, in_features, out_features, act=torch.nn.ReLU(inplace=True)):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features, bias=False)
    self.norm = torch.nn.BatchNorm1d(out_features)
    self.act = act

  def forward(self, input):
    output = self.linear(input)
    output = self.norm(output)
    output = self.act(output)
    return output


class MLP(torch.nn.Module):
  def __init__(self, in_features, out_features, hidden_features=256,
               hidden_layers=2, layer=FcBn, act=torch.nn.ReLU(inplace=True)):
    super().__init__()

    layers = [layer(in_features, hidden_features, act)]
    for _ in range(hidden_layers):
      layers.append(layer(hidden_features, hidden_features, act))
    layers.append(FC(hidden_features, out_features, act=torch.nn.Identity()))

    self.layers = torch.nn.Sequential(*layers)

  def forward(self, input):
    output = self.layers(input)
    return output
