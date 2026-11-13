"""Source-only single local Transformer; original query attention is untouched."""
import torch
from torch import nn
import torch.nn.functional as F
from experiments.native_receiver_runtime import forward_read as native_read

class MemoryContextWriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_norm=nn.LayerNorm(1024)
        self.stem=nn.Linear(1034,128)
        self.norm1=nn.LayerNorm(128);self.qkv=nn.Linear(128,384)
        self.attention_output=nn.Linear(128,128)
        self.norm2=nn.LayerNorm(128)
        self.ffn=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.relation=nn.Sequential(nn.Linear(18,64),nn.GELU(),nn.Linear(64,128,bias=False))
        self.final_norm=nn.LayerNorm(128);self.output=nn.Linear(128,1024)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,features,graph,kind):
        # API deliberately has no query/camera/id/target input.
        with torch.autocast('cuda',enabled=False):
            features=features.float();node=torch.cat((self.content_norm(features),graph['node10']),-1)
            x=self.stem(node);h=self.norm1(x);q,k,v=self.qkv(h).reshape(-1,3,4,32).unbind(1)
            neighbors=graph['neighbors'];mask=graph['neighbor_mask']
            extra=torch.zeros_like(graph['full4']) if kind=='P' else graph['sphere4' if kind=='S' else 'full4']
            rel=self.relation(torch.cat((graph['center14'],extra),-1)).reshape(-1,32,4,32)
            scores=((q[:,None]*k[neighbors]).sum(-1)+(q[:,None]*rel).sum(-1))/(32**.5)
            scores=scores.masked_fill(~mask[:,:,None],-torch.inf)
            attn=scores.softmax(1);context=(attn[:,:,:,None]*v[neighbors]).sum(1).reshape(-1,128)
            x=x+self.attention_output(context);x=x+self.ffn(self.norm2(x))
            result=features+self.output(self.final_norm(x))
            return torch.where(graph['valid'][:,None],result,features).half()

def forward_read(model,sample,fusion,writer=None,kind='P',checkpoint_blocks=True):
    changed=dict(sample)
    if writer is not None:
        encoded=writer(sample['source_features'],sample['graph'],kind)
        changed['extra']=sample['weights']@encoded.float()
    return native_read(model,changed,fusion,'Render',checkpoint_blocks=checkpoint_blocks)
