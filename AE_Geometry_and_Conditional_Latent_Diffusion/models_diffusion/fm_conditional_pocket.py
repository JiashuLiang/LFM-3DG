import torch
import torch.nn as nn
from zuko.utils import odeint
from .ddpm_conditional_pocket import MLP
from geometric_gnn_dojo_utils.models import EGNNModel
from torch_geometric.nn import radius_graph


class OTFlowMatching:
  ''' Optimal Transport Flow Matching
  '''
  
  def __init__(self, sig_min: float = 0.001) -> None:
    super().__init__()
    self.sig_min = sig_min
    self.eps = 1e-5

  def psi_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """ Conditional Flow
    """
    return (1 - (1 - self.sig_min) * t) * x + t * x_1

  def loss(self, v_t: nn.Module, x_1: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """ Compute loss
    """
    # t ~ Unif([0, 1])
    t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
    t = t[:, None].expand(x_1.shape)
    # x ~ p_t(x_0)
    x_0 = torch.randn_like(x_1) 
    v_psi = v_t(t[:,0], self.psi_t(x_0, x_1, t), condition)
    d_psi = x_1 - (1 - self.sig_min) * x_0
    return torch.mean((v_psi - d_psi) ** 2)
  


class VPDiffusionFlowMatching:
  ''' Variational Perserving Diffusion
  '''

  def __init__(self) -> None:
    super().__init__()
    self.beta_min = 0.1
    self.beta_max = 20.0
    self.eps = 1e-5

  def T(self, s: torch.Tensor) -> torch.Tensor:
    return self.beta_min * s + 0.5 * (s ** 2) * (self.beta_max - self.beta_min)

  def beta(self, t: torch.Tensor) -> torch.Tensor:
    return self.beta_min + t*(self.beta_max - self.beta_min)

  def alpha(self, t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * self.T(t))

  def mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    return self.alpha(1. - t) * x_1

  def sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1. - self.alpha(1. - t) ** 2)

  def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    num = torch.exp(-self.T(1. - t)) * x - torch.exp(-0.5 * self.T(1.-t))* x_1
    denum = 1. - torch.exp(- self.T(1. - t))
    return - 0.5 * self.beta(1. - t) * (num/denum)

  def loss(self, v_t: nn.Module, x_1: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """ Compute loss
    """ 
    # t ~ Unif([0, 1])
    t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
    t = t[:, None].expand(x_1.shape)
    # x ~ p_t(x|x_1)
    x = self.mu_t(t, x_1) + self.sigma_t(t, x_1) * torch.randn_like(x_1)

    return torch.mean((v_t(t[:,0], x, condition) - self.u_t(t, x, x_1)) ** 2)



class VEDiffusionFlowMatching:
  ''' Variational Exploding Diffusion
  '''

  def __init__(self) -> None:
    super().__init__()
    self.sigma_min = 0.01
    self.sigma_max = 2.
    self.eps = 1e-5


  def sigma_t(self, t: torch.Tensor) -> torch.Tensor:

    return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

  def dsigma_dt(self, t: torch.Tensor) -> torch.Tensor:

    return self.sigma_t(t) * torch.log(torch.tensor(self.sigma_max/self.sigma_min))

  def u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:

    return -(self.dsigma_dt(1. - t) / self.sigma_t(1. - t)) * (x - x_1)

  def loss(self, v_t: nn.Module, x_1: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """ Compute loss
    """ 
    # t ~ Unif([0, 1])
    t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) % (1 - self.eps)
    t = t[:, None].expand(x_1.shape)
    # x ~ p_t(x|x_1)
    x = x_1 + self.sigma_t(1. - t) * torch.randn_like(x_1)

    return torch.mean((v_t(t[:,0], x, condition) - self.u_t(t, x, x_1)) ** 2)
  
    

class CondVF(nn.Module):
  ''' Conditional Vector Field, used for sampling
    '''
  def __init__(self, net: nn.Module) -> None:
    super().__init__()
    self.net = net

    # This is conditional version
    def decode_t0_t1(self, x_0, t0, t1, condition):
        x_condition = self.condition_layer(condition)
        x_0 = torch.cat([x_0, x_condition], dim=1)
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())
    
    def encode(self, x_1: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_condition = self.condition_layer(condition)
        x_1 = torch.cat([x_1, x_condition], dim=1)
        return odeint(self.wrapper, x_1, 1., 0., self.parameters())
    
    def decode(self, x_0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_condition = self.condition_layer(condition)
        x_0 = torch.cat([x_0, x_condition], dim=1)
        return odeint(self.wrapper, x_0, 0., 1., self.parameters())
    


    
class Backbone(nn.Module):
    ''' Similar architecture to Diffusion Model to compare easily
     Only changed is time embedding method 
    '''
    def __init__(self, dim_in, dim_condition, dim_hidden, num_layer):
        super().__init__()
        self.dim_in = dim_in
        self.dim_condition = dim_condition
        self.dim_hidden = dim_hidden
        self.num_layer = num_layer
        
        self.linear_model1 = MLP(dim_in + dim_condition, dim_hidden, dim_hidden, num_layer)
        self.linear_model2 = MLP(dim_hidden, dim_hidden, dim_in, num_layer)

        # condition value
        self.condition_layer = nn.Linear(dim_condition, dim_condition)

    def time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        n_frequencies = dim // 2
        freq = 2 * torch.pi * torch.arange(n_frequencies, device=t.device)
        t = freq * t[..., None]
        embedding = torch.cat([torch.cos(t), torch.sin(t)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, x: torch.Tensor, condition=None):
        # For use in sampling
        if condition is None:
            assert x.shape[-1] == self.dim_in + self.dim_condition
            condition = x[..., self.dim_in:]
            x = x[..., :self.dim_in]
        t_embed = self.time_embedding(t, self.dim_hidden)
        x_condition = self.condition_layer(condition)
        x = self.linear_model2(self.linear_model1( torch.cat([x, x_condition], dim=1)) + t_embed)
        return x 
       

    
# Choose the model to use
def get_model(name: str):
    if name == "vp":
      return VPDiffusionFlowMatching()
    elif name == "ve":
      return VEDiffusionFlowMatching()
    if name == "ot":
      return OTFlowMatching()
    
class LFM_Cond(nn.Module):
    ''' Conditional Latent Flow Matching, analogous to LDM_Cond in molopt_score_model.py
    '''
    def __init__(self, config, model_type, protein_atom_feature_dim, ligand_atom_feature_dim, dim_in =500):
        super().__init__()
        self.encoder_3d = EGNNModel(num_layers=2, emb_dim=256, in_dim=27, out_dim=config.hidden_dim, aggr='mean', pool='mean')

        self.net = Backbone(dim_in=500, dim_condition=config.hidden_dim, dim_hidden=2048, num_layer=32)
        self.net = self.net.to('cuda')
        self.fm = get_model(model_type)
        self.condvf  = CondVF(self.net )   
        self.dim_in = dim_in
    
    def forward(
            self, protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand, emb, time_step=None, return_all=False, fix_x=False
    ):

        edge_index = radius_graph(protein_pos, r=10, batch=batch_protein)
        _, emb_protein = self.encoder_3d(protein_v, protein_pos, edge_index, batch_protein, tanh=False)

        loss = self.fm.loss(self.net, emb, emb_protein)

        return loss
    
    def sample_z(self, protein_pos, protein_v, batch_protein, num_sample, device="cuda"):
        edge_index = radius_graph(protein_pos, r=10, batch=batch_protein)
        _, emb_protein = self.encoder_3d(protein_v, protein_pos, edge_index, batch_protein, tanh=False)

        # repeat num_sample times for each protein
        _emb_protein = emb_protein.unsqueeze(dim=1).repeat((1, num_sample, 1)).reshape((emb_protein.shape[0]*num_sample, -1))
        n_samples = _emb_protein.shape[0]
        sample = torch.randn(n_samples, self.dim_in, device=device).squeeze()

        # split into 20 steps to clip sample to [-1, 1] 
        for idx in range(20):
            t0 = float(idx) / 20.0
            t1 = float(idx + 1) / 20.0
            sample = self.condvf.decode_t0_t1(sample, t0, t1, _emb_protein)
            sample[sample>1] = 1
            sample[sample<-1] = -1

        return sample 

