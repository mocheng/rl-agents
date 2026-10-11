import pytest

torch = pytest.importorskip("torch")

from rl_agents.agents.common.models import activation_factory, model_factory

ACTIVATIONS = ["RELU", "TANH", "LEAKY_RELU", "ELU"]


@pytest.mark.parametrize("activation_type", ACTIVATIONS)
def test_activation_factory(activation_type):
    activation = activation_factory(activation_type)
    x = torch.tensor([-2., -0.5, 0., 0.5, 2.])
    y = activation(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_unknown_activation():
    with pytest.raises(ValueError):
        activation_factory("UNKNOWN")


@pytest.mark.parametrize("activation_type", ACTIVATIONS)
def test_mlp_forward(activation_type):
    model = model_factory({"type": "MultiLayerPerceptron", "in": 4, "out": 2,
                           "layers": [8, 8], "activation": activation_type})
    y = model(torch.randn(3, 4))
    assert y.shape == (3, 2)


def test_dueling_network_activations():
    model = model_factory({"type": "DuelingNetwork", "in": 4, "out": 3,
                           "base_module": {"type": "MultiLayerPerceptron", "layers": [8],
                                           "activation": "LEAKY_RELU"},
                           "value": {"type": "MultiLayerPerceptron", "layers": [],
                                     "activation": "LEAKY_RELU"},
                           "advantage": {"type": "MultiLayerPerceptron", "layers": [],
                                         "activation": "LEAKY_RELU"}})
    y = model(torch.randn(3, 4))
    assert y.shape == (3, 3)


def test_negative_gradients_flow_through_leaky_relu_and_elu():
    x = torch.tensor([-1.0], requires_grad=True)
    activation_factory("RELU")(x).backward()
    assert x.grad.item() == 0

    x = torch.tensor([-1.0], requires_grad=True)
    activation_factory("LEAKY_RELU")(x).backward()
    assert x.grad.item() == pytest.approx(0.01)

    x = torch.tensor([-1.0], requires_grad=True)
    activation_factory("ELU")(x).backward()
    assert x.grad.item() > 0
