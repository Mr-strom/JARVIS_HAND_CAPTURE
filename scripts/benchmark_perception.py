"""Evaluate simple orange-object RGB segmentation against simulator segmentation."""
import sys,json
from pathlib import Path
import cv2,numpy as np,mujoco
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from manipulation import scene
from simulate import camera_for
results=[]
for offset in [-.008,0,.008]:
    m=scene(offset);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    with mujoco.Renderer(m,height=480,width=640) as r:
        r.update_scene(d,camera=camera_for(m));rgb=r.render().copy()
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask=(hsv[:,:,0]<25)&(hsv[:,:,1]>130)&(hsv[:,:,2]>80)
        r.enable_segmentation_rendering();r.update_scene(d,camera=camera_for(m));seg=r.render();truth=(seg[:,:,0]==m.geom('cylinder').id)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        union=(mask|truth).sum();intersection=(mask&truth).sum()
        center=lambda x:np.argwhere(x).mean(0) if x.any() else None
        a,b=center(mask),center(truth)
        results.append({'offset_m':offset,'iou':float(intersection/union) if union else None,'centroid_error_px':float(np.linalg.norm(a-b)) if a is not None and b is not None else None,'detected_pixels':int(mask.sum()),'truth_pixels':int(truth.sum()),'scope':'RGB orange segmentation, evaluated against segmentation IDs; no real-world depth or recognition claim'})
        if offset==0:cv2.imwrite(str(root/'private/object_perception.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
(root/'reports/perception_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
