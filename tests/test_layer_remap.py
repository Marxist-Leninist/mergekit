import pytest
import torch
from mergekit.layer_remap import remap_layers

def fixture():
    return {'emb.weight': torch.tensor([9.]), 'blocks.0.w': torch.tensor([1., 2.]), 'blocks.1.w': torch.tensor([3., 4.])}

def test_identity_no_alias():
    s=fixture(); out=remap_layers(s,[0,1])
    for k in s: assert torch.equal(s[k],out[k]) and s[k].data_ptr()!=out[k].data_ptr()

def test_permutation_and_repeated_donor():
    s=fixture(); out=remap_layers(s,[1,0]); assert torch.equal(out['blocks.0.w'],s['blocks.1.w'])
    out=remap_layers(s,[0,0]); out['blocks.0.w'][0]=77
    assert out['blocks.1.w'][0]==1 and s['blocks.0.w'][0]==1

@pytest.mark.parametrize('order',[[],[0],[0,2],[-1,0],[True,0],[0.,1]])
def test_invalid_order(order):
    with pytest.raises(ValueError): remap_layers(fixture(),order)

def test_shape_dtype_and_key_mismatch():
    for value in [torch.ones(3),torch.ones(2,dtype=torch.float64)]:
        s=fixture();s['blocks.1.w']=value
        with pytest.raises(ValueError):remap_layers(s,[1,0])
    s=fixture();s['blocks.1.extra']=torch.ones(1)
    with pytest.raises(ValueError):remap_layers(s,[1,0])

def test_custom_prefix_and_missing_layer():
    s={k.replace('blocks','encoder.layers'):v for k,v in fixture().items()}
    assert remap_layers(s,[1,0],prefix='encoder.layers')['encoder.layers.0.w'][0]==3
    with pytest.raises(ValueError):remap_layers(s,[0,1])
