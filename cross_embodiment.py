"""Held-out personal clips -> three Shadow pathways, evaluated in palm-width units.

The v4 adapter/controller are reused unchanged. Scratch FK uses equal splitting
of each coupled J0 tendon across J1/J2; physical results do not assume that split.
"""
import argparse, csv, json, time
import cv2, mujoco, numpy as np, torch
from humanoid_control import load_model, BimanualController, camera
from embodiment_adapter import ShadowGeometryAdapter
from learning.runtime import NeuralRetargeter, neural_features
from retargeting import Calibration, Retargeter
from simulate import ROOT, load_hand
from physics_guard import PhysicsGuard

DIGITS = ('ff', 'mf', 'rf', 'th')
METHODS = ('direct_rom', 'geometry_rom_leap', 'geometry_neural_leap')


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def human_geometry(canonical):
    """x thumb-side, y forward, z signed curl; scale = index/little MCP span."""
    ids = [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [1, 2, 3, 4]]
    p = np.asarray(canonical)[ids].copy()
    p -= p[:3, 0].mean(0)
    return p


class ShadowGeometry:
    def __init__(self, m):
        self.m = m
        self.palm = m.body('rh_palm').id
        self.bodies = [[m.body('rh_'+f+n).id for n in ('proximal', 'middle', 'distal')] for f in DIGITS]
        self.scratch = mujoco.MjData(m)
        mujoco.mj_forward(m, self.scratch)
        self.width = np.linalg.norm(self.scratch.xpos[m.body('rh_ffknuckle').id] - self.scratch.xpos[m.body('rh_lfknuckle').id])
        # Distal axial endpoints from the actual compiled meshes, not arbitrary sites.
        self.tip_offsets = []
        for chain in self.bodies:
            b = chain[-1]
            gid = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
            mesh = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            world = v @ self.scratch.geom_xmat[gid].reshape(3,3).T + self.scratch.geom_xpos[gid]
            local = (world-self.scratch.xpos[b]) @ self.scratch.xmat[b].reshape(3,3)
            self.tip_offsets.append(np.array([0., 0., local[:,2].max()]))

    def points(self, d):
        r = d.xmat[self.palm].reshape(3,3)
        points = []
        for bodies, offset in zip(self.bodies, self.tip_offsets):
            chain = [d.xpos[b].copy() for b in bodies]
            chain.append(d.xpos[bodies[-1]] + d.xmat[bodies[-1]].reshape(3,3) @ offset)
            p = (np.array(chain)-d.xpos[self.palm]) @ r
            points.append(p[:, [0,2,1]] * np.array([1.,1.,-1.]))
        p = np.array(points)/self.width
        p -= p[:3,0].mean(0)
        return p

    def kinematic(self, command):
        d = self.scratch
        mujoco.mj_resetData(self.m, d)
        for a in range(self.m.nu):
            if not self.m.actuator(a).name.startswith('rh_'): continue
            if self.m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                j = self.m.actuator_trnid[a,0]
                d.qpos[self.m.jnt_qposadr[j]] = command[a]
            else:
                name = self.m.actuator(a).name[5:7]
                for suffix in ('J1', 'J2'):
                    j = self.m.joint('rh_'+name+suffix).id
                    d.qpos[self.m.jnt_qposadr[j]] = np.clip(command[a]/2, *self.m.jnt_range[j])
        mujoco.mj_forward(self.m, d)
        return self.points(d)


def geometry_metrics(p, h):
    tips, ht = p[:,3], h[:,3]
    return {
        'fingertip_error_palm_widths': float(np.linalg.norm(tips-ht,axis=-1).mean()),
        'finger_direction_cosine': float(np.sum(unit(tips-p[:,0])*unit(ht-h[:,0]),axis=-1).mean()),
        'pinch_error_palm_widths': float(np.abs(np.linalg.norm(tips[:3]-tips[3],axis=-1)-np.linalg.norm(ht[:3]-ht[3],axis=-1)).mean())}


class Pathways:
    def __init__(self, m):
        self.m = m
        leap = load_hand()
        names = [leap.joint(int(j)).name for j in leap.actuator_trnid[:,0]]
        self.cal = Calibration.load(ROOT/'private/calibration.json')
        self.rom = Retargeter(names, leap.actuator_ctrlrange, self.cal)
        self.neural = NeuralRetargeter(names, leap.actuator_ctrlrange, self.cal)
        self.adapter = ShadowGeometryAdapter(m)
        self.neutral = np.clip(np.zeros(m.nu),m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        for a in range(m.nu):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and not m.actuator(a).name.startswith(('rh_', 'lh_')):
                self.neutral[a] = m.qpos0[m.jnt_qposadr[m.actuator_trnid[a,0]]]

    def command(self, method, world):
        x = neural_features(world)
        q = self.rom.map_features(x[63:]) if method != 'geometry_neural_leap' else self.neural.map_features(x)
        command = self.neutral.copy()
        if method == 'direct_rom':
            n = np.clip((x[63:]-self.cal.low)/(self.cal.high-self.cal.low),0,1)
            values = {}
            for k, digit in enumerate(('FF','MF','RF')):
                values[digit+'J3'] = 1.2*n[k*4]
                values[digit+'J0'] = 1.6*(n[k*4+2]+n[k*4+3])/2
                values[digit+'J4'] = -.35*(2*n[k*4+1]-1)
            values.update(THJ4=.8*n[12], THJ5=.8*n[13], THJ1=.8*(n[14]+n[15])/2)
            mapped = {self.m.actuator('rh_A_'+name).id:value for name,value in values.items()}
        else:
            mapped = self.adapter.commands(q)
        for a,v in mapped.items(): command[a] = v
        return np.clip(command,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])


def evaluate(render=False, dataset=None):
    out = ROOT/'private/final';out.mkdir(parents=True,exist_ok=True)
    base = ROOT if dataset is None else __import__('pathlib').Path(dataset).resolve()
    audit = json.loads((ROOT/'reports/dataset_audit.json' if dataset is None else base/'manifest.json').read_text())
    clips = [e for e in audit['split'] if e['partition']=='test']
    m = load_model(); mapper = Pathways(m); geom = ShadowGeometry(m)
    right = BimanualController(m).hands['right']
    joints = np.array([j for j in range(m.njnt) if m.joint(j).name.startswith('rh_')]);qa=m.jnt_qposadr[joints]
    results=[];trajectories={};render_id=max(clips,key=lambda e:e['valid_frames'])['index']
    renderer=mujoco.Renderer(m,height=480,width=640) if render else None
    video=cv2.VideoWriter(str(out/'cross_embodiment.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,960)) if render else None
    if video is not None and not video.isOpened():raise RuntimeError('Cannot encode transfer video')
    try:
        for entry in clips:
            z=np.load(base/entry['cache']);keep=z['valid'].astype(bool)
            worlds=z['world'][keep];human=z['canonical'][keep];times=z['timestamps'][keep];indices=z['frame_indices'][keep]
            states={method:mujoco.MjData(m) for method in METHODS}
            controllers={method:BimanualController(m) for method in METHODS}
            guards={method:PhysicsGuard(m,d) for method,d in states.items()}
            previous={method:mapper.neutral.copy() for method in METHODS}
            for c in controllers.values():c.command[:]=mapper.neutral
            stats={method:{'rows':[],'latency':[],'tracking':[],'raw_limit_samples':0,'max_excursion_rad':0.,'max_penetration_m':0.} for method in METHODS}
            for d in states.values():mujoco.mj_forward(m,d)
            if render and not entry.get('source'):raise ValueError('Portable landmark-only pack has no RGB source; omit --render')
            cap=cv2.VideoCapture(entry['source']) if render and entry['index']==render_id else None
            cache=[];decoded=-1;source=None
            for n,(world,h,t) in enumerate(zip(worlds,human,times)):
                goal=human_geometry(h);frames=[]
                # Respect actual timestamps, including missing-frame intervals. Initial pose receives no free settling.
                elapsed=0. if n==0 else t-times[n-1]
                for method in METHODS:
                    d=states[method];c=controllers[method];g=guards[method];s=stats[method]
                    tick=time.perf_counter();cmd=mapper.command(method,world);s['latency'].append((time.perf_counter()-tick)*1000)
                    kinematic=geom.kinematic(cmd)
                    target_time=float(t-times[0])
                    while d.time+1e-10<target_time:
                        c.step(d,previous[method]);g.step(d)
                        exc=np.maximum(m.jnt_range[joints,0]-d.qpos[qa],d.qpos[qa]-m.jnt_range[joints,1])
                        s['raw_limit_samples']+=int(np.count_nonzero(exc>1e-6));s['max_excursion_rad']=max(s['max_excursion_rad'],float(exc.max()))
                        s['max_penetration_m']=max(s['max_penetration_m'],max([0.]+[-float(ct.dist) for ct in d.contact]))
                    actual=geom.points(d);metrics=geometry_metrics(actual,goal)
                    previous[method]=cmd.copy()
                    if np.any(cmd<m.actuator_ctrlrange[:,0]) or np.any(cmd>m.actuator_ctrlrange[:,1]):raise RuntimeError('Command outside limits')
                    row={'recording_id':entry['index'],'method':method,'timestamp_s':float(t),**metrics}
                    row.update({'kinematic_'+k:v for k,v in geometry_metrics(kinematic,goal).items()});s['rows'].append(row)
                    s['tracking'].append((d.actuator_length[right]-c.command[right])/(m.actuator_ctrlrange[right,1]-m.actuator_ctrlrange[right,0]))
                    cache.append((method,n,d.qpos.copy()))
                    if cap is not None and n<144:
                        cam=camera(m,True);cam.lookat[:]=d.xpos[m.body('rh_palm').id];cam.distance=.44
                        renderer.update_scene(d,camera=cam);f=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                        cv2.putText(f,method,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                        cv2.putText(f,f't={t:.2f}s  normalized tip error={metrics["fingertip_error_palm_widths"]:.3f}',(12,454),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);frames.append(f)
                if frames:
                    while decoded<int(indices[n]):
                        ok,source=cap.read();decoded+=1
                        if not ok:raise RuntimeError('Source video decode failed')
                    f=cv2.resize(source,(640,480));cv2.putText(f,'Same held-out personal frame / fixed arms',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                    video.write(np.vstack([np.hstack([f,frames[0]]),np.hstack(frames[1:])]))
                if n%150==0:print('clip',entry['index'],'frame',n,'/',len(times),flush=True)
            if cap:cap.release()
            trajectories[str(entry['index'])]=cache
            for method,s in stats.items():
                results.append({'recording_id':entry['index'],'method':method,'frames':len(times),'sim_seconds':float(states[method].time),
                    **{k:float(np.mean([row[k] for row in s['rows']])) for k in s['rows'][0] if k not in ('recording_id','method','timestamp_s')},
                    'mapping_latency_mean_ms':float(np.mean(s['latency'][1:])),'mapping_latency_p95_ms':float(np.percentile(s['latency'][1:],95)),
                    'dynamic_tracking_rmse_actuator_range_fraction':float(np.sqrt(np.mean(np.array(s['tracking'])**2))),
                    'raw_joint_limit_samples_over_1e-6_rad':s['raw_limit_samples'],'max_joint_limit_excursion_rad':max(0.,s['max_excursion_rad']),
                    'max_penetration_m':s['max_penetration_m'],'warnings':guards[method].warnings(states[method]),'command_limit_violations':0})
            # Private per-frame evidence, never exposed in the public aggregate.
            (out/f'cross_clip_{entry["index"]}.json').write_text(json.dumps([row for s in stats.values() for row in s['rows']]))
    finally:
        if renderer:renderer.close();video.release()
    summary={}
    for method in METHODS:
        rows=[r for r in results if r['method']==method];keys=[k for k in rows[0] if k not in ('recording_id','method','warnings')]
        summary[method]={k:float(np.mean([r[k] for r in rows])) for k in keys}
        for k in ['frames','sim_seconds','raw_joint_limit_samples_over_1e-6_rad','command_limit_violations']:summary[method][k]=sum(r[k] for r in rows)
        for k in ['max_joint_limit_excursion_rad','max_penetration_m']:summary[method][k]=max(r[k] for r in rows)
        summary[method]['warning_count']=sum(sum(r['warnings'].values()) for r in rows)
    report={'protocol':'five unchanged held-out recordings; equal-weight recording means; no held-out tuning; original timestamps, zero-order held targets; no object; fixed base/arms; all right hand joints and contacts checked each step',
        'coordinates':'Human canonical x=index/thumb side, y=wrist-to-middle projected forward, z=cross(x,y). Shadow local -> [x,z,-y]. Center at mean index/middle/ring MCP. Each hand divided by its own index/little MCP span; dimensions are palm widths, not millimetres.',
        'endpoints':'Shadow distal local +z mesh axial endpoint proxy; human MediaPipe fingertips. Direction = base-to-tip cosine. Pinch = thumb-to-index/middle/ring distances. No external ground truth.',
        'pathways':{'direct_rom':'calibrated human geometry directly to named Shadow ROM; fixed thumb heuristic','geometry_rom_leap':'Phase2 calibrated ROM -> LEAP FK -> unchanged v4 Shadow adapter','geometry_neural_leap':'actual trained personalized MLP -> LEAP FK -> same unchanged v4 adapter'},
        'latency_scope':'CPU world-landmarks to bounded Shadow commands, including feature extraction/MLP/FK as applicable; excludes camera, MediaPipe, simulation, filtering, rendering, display. Not physical end-to-end motion latency.',
        'kinematic_scope':'scratch FK command reference, tendon J0 split equally into J1/J2; dynamic metrics use actual physical joint states without this assumption',
        'statistics':'five recording means, not independent-person trials; unequal valid-frame counts; no significance/general-purpose generalization claim',
        'summary':summary,'per_recording':results}
    (ROOT/'reports/cross_embodiment_results.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'reports/cross_embodiment.csv').open('w',newline='') as f:
        rows=[{k:v for k,v in r.items() if k!='warnings'} for r in results];w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--render',action='store_true');p.add_argument('--dataset');a=p.parse_args();evaluate(a.render,a.dataset)
