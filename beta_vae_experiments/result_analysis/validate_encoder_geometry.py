"""Validate actual intrinsic curvature on known smooth surfaces, without a VAE."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..metrics.complementary.curvature.encoder_geometry import immersion_curvature


def validate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(31)
    surfaces={
        'plane':(lambda t:torch.stack((t[0],t[1],0*t[0])),lambda t:0.),
        'cylinder':(lambda t:torch.stack((torch.cos(t[0]),torch.sin(t[0]),t[1])),lambda t:0.),
        'swiss_roll':(lambda t:torch.stack(((t[0]+3)*torch.cos(t[0]+3),t[1],(t[0]+3)*torch.sin(t[0]+3))),lambda t:0.),
        'saddle':(lambda t:torch.stack((t[0],t[1],(t[0]**2-t[1]**2)/2)),lambda t:-2/(1+float(t.square().sum()))**2),
    }
    for radius in (1.,2.):
        surfaces[f'sphere_R{radius:g}']=(lambda t,r=radius:r*torch.stack((torch.cos(t[0])*torch.cos(t[1]),torch.sin(t[0])*torch.cos(t[1]),torch.sin(t[1]))),lambda t,r=radius:2/r**2)
    rows=[]
    for name,(surface,truth) in surfaces.items():
        for point in rng.uniform(-.6,.6,size=(5,2)):
            point=torch.tensor(point,dtype=torch.float64)
            result=immersion_curvature(surface,point,smooth=True)
            expected=truth(point)
            error=abs(result['scalar_curvature']-expected)
            assert result['status']=='ok' and error<1e-9,(name,result,expected)
            rows.append(dict(surface=name,u=float(point[0]),v=float(point[1]),expected_scalar=expected,
                             absolute_error=error,**result))
    result=pd.DataFrame(rows)
    result.to_csv(output/'known_surface_curvature_validation.csv',index=False)
    print(result.groupby('surface').absolute_error.max().to_string())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    validate(parser.parse_args().output)
