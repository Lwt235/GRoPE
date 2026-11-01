"""Small deterministic analytic renderer: textured quads and cuboid faces.

Pixels are cast independently through calibrated pinhole cameras. The nearest
positive quad intersection determines RGB, depth, normal and surface id. Surface
ids are used solely for label/visibility generation and never model inputs.
"""
from pathlib import Path
import numpy as np
from PIL import Image

WIDTH, HEIGHT, PATCH, FOCAL = 96, 72, 12, 75.
COLS, ROWS = WIDTH//PATCH, HEIGHT//PATCH


def camera(center, target=(0.,0.,5.)):
    center=np.asarray(center,dtype=np.float64)
    forward=np.asarray(target)-center
    forward/=np.linalg.norm(forward)
    right=np.cross(np.array([0.,1.,0.]),forward)
    right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.stack([right,down,forward])
    return rotation,-rotation@center,center


def scene(seed):
    rng=np.random.default_rng(seed)
    # center, tangent u, tangent v, half-width, half-height
    faces=[(np.array([0.,0.,7.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]),20.,20.)]
    # Three boxes yield surfaces with shared normals over many patches.
    for index in range(3):
        center=np.array([(-.9+index*.9)+rng.uniform(-.2,.2),rng.uniform(-.4,.5),rng.uniform(3.3,5.4)])
        half=np.array([rng.uniform(.3,.65),rng.uniform(.4,.9),rng.uniform(.25,.6)])
        yaw=rng.uniform(-.65,.65)
        rot=np.array([[np.cos(yaw),0.,np.sin(yaw)],[0.,1.,0.],[-np.sin(yaw),0.,np.cos(yaw)]])
        for axis in range(3):
            other=[i for i in range(3) if i!=axis]
            for sign in [-1,1]:
                faces.append((center+sign*half[axis]*rot[:,axis],rot[:,other[0]],rot[:,other[1]],half[other[0]],half[other[1]]))
    # Repeated world-stationary texture, no per-token random identifier.
    palette=rng.uniform(.25,.8,(4,3))
    properties=[]
    for index in range(len(faces)):
        properties.append({'palette':palette[index%4], 'phase':rng.uniform(0,2*np.pi,2),
                           'frequency':rng.uniform(2.,5.,2)})
    return faces,properties,rng


def render(faces,properties,cam):
    rotation,translation,origin=cam
    yy,xx=np.mgrid[:HEIGHT,:WIDTH]
    rays_cam=np.stack([(xx-WIDTH/2)/FOCAL,(yy-HEIGHT/2)/FOCAL,np.ones_like(xx)],-1).reshape(-1,3)
    rays=rays_cam@rotation
    count=len(rays)
    best=np.full(count,np.inf)
    surface=np.full(count,-1,dtype=np.int64)
    local=np.zeros((count,2))
    normals=np.zeros((count,3))
    for index,(center,u,v,su,sv) in enumerate(faces):
        normal=np.cross(u,v)
        denominator=(rays*normal).sum(-1)
        distance=((center-origin)*normal).sum()
        depth=np.divide(distance,denominator,out=np.full(count,np.inf),where=np.abs(denominator)>1e-9)
        point=origin+rays*np.where(np.isfinite(depth),depth,0.)[:,None]
        displacement=point-center
        du,dv=(displacement*u).sum(-1),(displacement*v).sum(-1)
        mask=(depth>.05)&(depth<best)&(np.abs(du)<=su)&(np.abs(dv)<=sv)
        best[mask]=depth[mask]
        surface[mask]=index
        local[mask]=np.stack([du[mask],dv[mask]],-1)
        normals[mask]=normal
    assert np.isfinite(best).all(), 'Background must fill the image.'
    rgb=np.zeros((count,3))
    light=np.array([.3,-.4,-.85])
    light/=np.linalg.norm(light)
    for index,prop in enumerate(properties):
        mask=surface==index
        uv=local[mask]
        freq=prop['frequency']
        phase=prop['phase']
        stripe=np.sin(uv[:,0]*freq[0]*2*np.pi+phase[0])*np.sin(uv[:,1]*freq[1]*2*np.pi+phase[1])
        checker=np.sign(np.sin(uv[:,0]*freq[0]*np.pi)*np.sin(uv[:,1]*freq[1]*np.pi))
        intensity=.8+.15*stripe+.1*checker
        diffuse=.75+.25*np.abs(normals[mask]@light)
        rgb[mask]=np.clip(prop['palette']*intensity[:,None]*diffuse[:,None],0,1)
    points=origin+rays*best[:,None]
    return {'rgb':rgb.reshape(HEIGHT,WIDTH,3).astype(np.float32),
            'depth':best.reshape(HEIGHT,WIDTH).astype(np.float32),
            'normal':normals.reshape(HEIGHT,WIDTH,3).astype(np.float32),
            'points':points.reshape(HEIGHT,WIDTH,3).astype(np.float32),
            'surface':surface.reshape(HEIGHT,WIDTH),
            'rotation':rotation.astype(np.float32),'translation':translation.astype(np.float32)}


def pair(seed,wide=False):
    faces,properties,rng=scene(seed)
    first=camera([0.,0.,0.])
    baseline=rng.uniform(.45,.95)*(1.8 if wide else 1.)
    second=camera([baseline,rng.uniform(-.12,.12),rng.uniform(-.1,.1)])
    a,b=render(faces,properties,first),render(faces,properties,second)
    centers_x=np.arange(COLS)*PATCH+PATCH//2
    centers_y=np.arange(ROWS)*PATCH+PATCH//2
    cy,cx=np.meshgrid(centers_y,centers_x,indexing='ij')
    cx,cy=cx.flatten(),cy.flatten()
    query_points=a['points'][cy,cx]
    target_points=query_points@b['rotation'].T+b['translation']
    uv=target_points[:,:2]/target_points[:,2:]*FOCAL+np.array([WIDTH/2,HEIGHT/2])
    inside=(uv[:,0]>=0)&(uv[:,0]<WIDTH)&(uv[:,1]>=0)&(uv[:,1]<HEIGHT)&(target_points[:,2]>.05)
    pixel=np.rint(uv).astype(int)
    px=np.clip(pixel[:,0],0,WIDTH-1)
    py=np.clip(pixel[:,1],0,HEIGHT-1)
    visible=inside&(np.abs(target_points[:,2]-b['depth'][py,px])<.06)&(a['surface'][cy,cx]==b['surface'][py,px])
    tx=np.clip((uv[:,0]/PATCH).astype(int),0,COLS-1)
    ty=np.clip((uv[:,1]/PATCH).astype(int),0,ROWS-1)
    target=ty*COLS+tx
    # A point is representable by a patch only if the patch center is on the
    # same physical surface. Boundary/mixed-support failures become unmatched.
    represented=b['surface'][centers_y[ty],centers_x[tx]]==a['surface'][cy,cx]
    match=visible&represented
    labels=np.where(match,target,COLS*ROWS).astype(np.int64)
    return a,b,{'labels':labels,'center_x':cx,'center_y':cy,
                 'visible':visible,'represented':represented,'target_uv':uv.astype(np.float32)}


def preview(seed=600001,out='results/round2_rendered/preview.png'):
    a,b,gt=pair(seed)
    images=[]
    for view in [a,b]:
        rgb=(view['rgb']*255).astype(np.uint8)
        normal=((view['normal']+1)*127.5).astype(np.uint8)
        depth=np.clip((view['depth']-2)/6,0,1)
        depth=(np.stack([depth,depth,depth],-1)*255).astype(np.uint8)
        images.append(np.concatenate([rgb,depth,normal],1))
    canvas=Image.fromarray(np.concatenate(images,0)).resize((WIDTH*9,HEIGHT*6))
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out)
    return out


if __name__=='__main__':
    print(preview())
