import numpy as np,torch,pytest
from v6.geometry import Geometry
from v6.adaptive import AdaptiveEngine,MotionSafety,Decision
from learning.leap_fk import LeapFK
from learning.runtime import neural_features
from train_personal import targets
from simulate import ROOT

personal_required=pytest.mark.skipif(not (ROOT/'private/learning/personal_mlp.pt').exists() or not (ROOT/'private/calibration.json').exists(),reason='Personal checkpoint/calibration are private and require approval')

def hand(curl=.3):
    p=np.zeros((21,3));p[1]=[.035,.018,0];p[2]=[.06,.035,.005];p[3]=[.075,.055,.005];p[4]=[.085,.07,.005]
    for start,x in [(5,.03),(9,0),(13,-.025),(17,-.05)]:
        p[start]=[x,.065,0]
        for k,length in enumerate([.035,.025,.02],1):p[start+k]=p[start+k-1]+[0,length*np.cos(curl*k),length*np.sin(curl*k)]
    return p

def test_analytical_fk_jacobian_and_original_objective():
    g=Geometry();q=(g.low+g.high)/2;goal=g.targets(neural_features(hand())[:63].reshape(21,3));r,j=g.residual_jac(q,goal)
    numerical=[]
    for k in range(16):
        a=q.copy();b=q.copy();a[k]+=1e-6;b[k]-=1e-6;numerical.append((g.residual_jac(a,goal,False)[0]-g.residual_jac(b,goal,False)[0])/2e-6)
    np.testing.assert_allclose(j,np.array(numerical).T,atol=1e-7)
    fk=LeapFK()
    with torch.no_grad():p=fk(torch.tensor(q[None],dtype=torch.float32)).numpy()[0]
    np.testing.assert_allclose(g.points_jac(q,False)[0],p[3::4],atol=1e-7)
    from train_personal import objective
    expected=objective(torch.tensor(p[None]),torch.tensor(goal[None],dtype=torch.float32)).item()
    assert r@r/12==pytest.approx(expected,rel=1e-5)
    np.testing.assert_allclose(goal,targets(fk,np.array([neural_features(hand())[:63].reshape(21,3)])).numpy()[0],atol=1e-7)

def test_solver_never_worsens_and_deadline_is_reported():
    g=Geometry();q=(g.low+g.high)/2;goal=g.targets(neural_features(hand(.5))[:63].reshape(21,3))
    fit,info=g.solve(goal,q,budget_ms=0,max_nfev=8)
    assert info['timeout'];np.testing.assert_allclose(fit,q)
    fit,info=g.solve(goal,q,budget_ms=100,max_nfev=6)
    assert info['final_cost']<=info['initial_cost']+1e-12
    assert np.all(fit>=g.low) and np.all(fit<=g.high)

@personal_required
def test_unreliable_data_never_reaches_optimizer():
    engine=AdaptiveEngine('optimizer')
    def forbidden(*a,**k):raise AssertionError('Invalid observation was optimized')
    engine.geometry.solve=forbidden
    for x in [None,np.full(79,np.nan),np.zeros(79)]:assert engine.observe(x,1).mode=='RECOVERY'
    assert engine.observe(neural_features(hand()),1,now=2).reason=='stale_or_future'

@personal_required
def test_timeout_uses_personal_neural_prediction():
    engine=AdaptiveEngine('optimizer',budget_ms=0);x=neural_features(hand());d=engine.observe(x,0.)
    assert d.mode=='NEURAL' and d.reason=='solver_deadline_fallback'
    np.testing.assert_allclose(d.target,engine.personal.predict(x))

def test_safety_limits_hold_return_and_smooth_reacquisition():
    limits=np.array([[0.,1.],[-.35,.35]]);s=MotionSafety(limits);vel=[];q=[]
    for i in range(240):
        t=i/30;decision=Decision(np.array([1.,.35]),'PRECISION','test',t,{}) if i<60 or i>=180 else None
        q.append(s.update(decision,t));vel.append(s.velocity.copy())
        assert np.all(q[-1]>=limits[:,0]-1e-8) and np.all(q[-1]<=limits[:,1]+1e-8)
        if i==150:assert max(abs(q[-1]-s.neutral))<.02
    assert np.max(abs(np.array(vel)))<=s.max_speed+1e-8
    assert np.max(abs(np.diff(vel,axis=0))*30)<=s.max_acceleration+1e-7
    assert np.max(abs(np.diff(q,axis=0))*30)<=s.max_speed+1e-7
    with pytest.raises(ValueError):s.update(None,0)

@personal_required
def test_mode_hysteresis_dwell_and_no_direct_state_reset():
    e=AdaptiveEngine('hybrid');x=neural_features(hand(.3))
    e.geometry.solve=lambda goal,q,*a: (q,{'timeout':False,'initial_cost':1.,'final_cost':.9,'elapsed_ms':0.})
    # Force unambiguous slow high-error entry through an ordinary valid real-shaped hand.
    d=e.observe(x,0.);first=e.selected
    for t in [.01,.02,.03]:
        d=e.observe(x,t);assert e.selected==first
    assert e.switches<=1
