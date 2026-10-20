def test_cpu_best_state_is_an_independent_snapshot():
    import torch
    from backend.pipeline import LEGACY
    from train_layoutlmv3 import cpu_state_dict
    model = torch.nn.Linear(2, 2)
    before = cpu_state_dict(model)
    original = before['weight'].clone()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(before['weight'], original)
    assert not torch.equal(before['weight'], model.weight)
