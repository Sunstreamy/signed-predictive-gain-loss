import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import te_replication as t


class TEReplicationTests(unittest.TestCase):
    def test_run_lags_and_window_alignment(self):
        runs=np.array([np.arange(100),10000+np.arange(100)],dtype=float)[:,:,None]
        common=dict(mean=np.zeros(1),sd=np.ones(1))
        for raw in runs:
            lag=t.lag_run(raw,common)
            np.testing.assert_array_equal(lag[0],raw[[16,15,12,0],0])
            window=t.rolling_mean(lag[:,0],64)
            self.assertEqual(len(t.confirm(window)),len(raw)-81)
            self.assertEqual(window[0],raw[16:80,0].mean())
        samples,ix=t.sampled_cohort(runs,[1,2],common,np.random.default_rng(10),20)
        expected=np.concatenate([t.lag_run(raw,common) for raw in runs])[ix]
        np.testing.assert_array_equal(samples,expected)

    def test_masked_self_and_direct_forward(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(7,12))
        model=t.dev.PairedMLP(3,32,10)
        fast=model.predict_all(x,'self')
        for i in range(3):
            targets=np.full(len(x),i); direct=model.forward(model.inputs(x,targets,'self'))[0][:,i]
            np.testing.assert_allclose(fast[:,i],direct,atol=1e-13)
            changed=x.copy(); changed[:,i]+=100; changed[:,np.arange(12)%3!=i]+=200
            np.testing.assert_array_equal(fast[:,i],model.predict_all(changed,'self')[:,i])

    def test_ridge_projection_and_signed_identity(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(200,12)); r=rng.normal(size=(200,3))
        cfg=dict(ridge_lambda=.01,projection_rcond=1e-10,feature_variance_floor=1e-12,chunk=4096)
        model=t.residual.fit_correction(x,r,cfg)
        c=t.residual.predict_correction(x,model)
        loss=t.residual.paired_losses(r,c)
        np.testing.assert_allclose(loss[:,:,2],loss[:,:,1]-loss[:,:,0],atol=1e-14)
        for i in range(3):
            own,cross=t.residual.feature_indices(i,3)
            projected=(x[:,cross]-model['mean'][cross])-(x[:,own]-model['mean'][own])@model['projection'][i]
            np.testing.assert_allclose(c[:,i],projected@model['cross_beta'][i],atol=1e-14)
            self.assertEqual(model['weights'][i,i],0)

    def test_qualification_boundary_and_block_reset(self):
        cfg=json.loads(t.CONFIG.read_text()); cfg['source']['columns']=['a','b']
        means=np.array([[1.,.95,-.05],[1.,.95,-.05]])
        stages={k:(means.copy(),np.array([6,6]),10) for k in ['cross_A','cross_B','pooled_A','pooled_B']}
        self.assertTrue(all(r['qualified'] for r in t.qualify(stages,cfg)))
        stages['pooled_B'][0][1,2]=-.049999
        self.assertEqual([r['qualified'] for r in t.qualify(stages,cfg)],[True,False])
        a=np.zeros((421,2,3)); a[:,:,2]=-.1
        self.assertEqual(t.gain_statistics(a)[2],22)
        self.assertEqual(t.gain_statistics(np.concatenate([a,a]))[2],43)

    def test_calibration_ties(self):
        cal=[np.repeat(np.arange(100),3),np.arange(350)/3]
        v=np.r_[np.arange(400)/3,cal[0],cal[1]]
        direct=np.array([max((1+(c>=x).sum())/(len(c)+1) for c in cal) for x in v])
        np.testing.assert_array_equal(direct,t.tail(v,cal))
        for row in t.cutoffs(cal,t.ALPHAS):
            np.testing.assert_array_equal(direct<=row['alpha'],v>row['strict_upper_cutoff'])

    def test_onsets_preserve_prealarm_and_physical_units(self):
        f=np.zeros(960-81,bool); f[159-81:170-81]=True
        early=t.event_metrics(f,81,stop=240)
        self.assertFalse(early['detected']); self.assertTrue(early['pre_alarm']); self.assertTrue(early['any_hit'])
        f[180-81:183-81]=True
        full=t.event_metrics(f,81)
        self.assertEqual(full['delay_minutes'],60)
        self.assertEqual(full['alarm_episodes'],1)
        self.assertEqual(full['longest_alarm_minutes'],30)
        self.assertEqual(t.flag_metrics(f,81)['alarm_episodes'],2)


if __name__=='__main__': unittest.main()
