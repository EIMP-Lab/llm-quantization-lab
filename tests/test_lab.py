"""验证课堂数字、低位宽打包与模型回读；无需 pytest。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import tempfile
import numpy as np
import lab

class NumericalChecks(unittest.TestCase):
    def test_classroom_answer(self):
        rows=lab.gradient_descent()
        np.testing.assert_allclose([[r['a'],r['b'],r['loss']] for r in rows],
            [[0,0,2.5],[1.6,.8,.1],[1.92,.96,.004],[1.984,.992,.00016]],atol=1e-12)

    def test_compensation_is_minimizer(self):
        result=lab.toy_compensation();delta=result['delta']
        x0=np.array([1.,2.,3.]);x1=np.array([1.,2.1,2.9])
        loss=lambda z:np.sum((.03*x0-z*x1)**2)
        self.assertLess(loss(delta),loss(delta+.001))
        self.assertLess(loss(delta),loss(delta-.001))
        self.assertLess(result['compensated_sse'],result['rtn_sse']/100)

    def test_bitpacking_with_padding(self):
        rng=np.random.default_rng(71)
        for bits in (2,4,8):
            qmax=2**(bits-1)-1
            q=rng.integers(-qmax,qmax+1,(13,7),dtype=np.int8)
            raw=lab.pack_codes(q,bits)
            self.assertEqual(len(raw),(q.size*bits+7)//8)
            np.testing.assert_array_equal(lab.unpack_codes(raw,bits,q.shape),q)

    def test_zero_and_rank_deficient_inputs(self):
        w=np.zeros((8,3),dtype=np.float32);x=np.ones((4,8),dtype=np.float32)
        for bits in (2,4,8):
            q,c,s=lab.quantize(w,bits)
            self.assertTrue(np.isfinite(q).all())
            q,c,s=lab.gptq_educational(w,x,bits)
            np.testing.assert_array_equal(q,0)

    def test_algorithms_do_not_modify_baseline(self):
        rng=np.random.default_rng(3);w=rng.normal(size=(8,3)).astype('float32');old=w.copy();x=rng.normal(size=(20,8))
        for fn in (lab.gptq_educational,lab.awq_inspired):
            fn(w,x,4);np.testing.assert_array_equal(w,old)

    def test_model_packing_prediction_roundtrip(self):
        with np.load(lab.ROOT/'baseline.npz') as f:m={k:f[k] for k in f.files}
        d=lab.load_data();codes={};scales={};qm={k:v.copy() for k,v in m.items()}
        for k in ('w1','w2'):qm[k],codes[k],scales[k]=lab.quantize(m[k],4)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'weights.qbin';lab.save_packed(p,qm,codes,scales,4)
            decoded=lab.load_packed(p)
            np.testing.assert_allclose(lab.forward(decoded,d['x_test']),lab.forward(qm,d['x_test']),atol=1e-5)
        self.assertGreater(lab.accuracy(m,d['x_test'],d['y_test']),.90)

if __name__=='__main__':unittest.main(verbosity=2)
