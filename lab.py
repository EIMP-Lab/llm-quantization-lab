"""CPU 量化实训：只需 NumPy；全部数据、模型在本目录。"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_data():
    """使用 UCI 官方 train/test 划分；只在 train 内分出校准和验证。"""
    train = np.loadtxt(ROOT / 'data/optdigits.tra', delimiter=',')
    test = np.loadtxt(ROOT / 'data/optdigits.tes', delimiter=',')
    rng = np.random.default_rng(42)
    subsets = [[], [], []]
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(train[:, -1] == c))
        n1, n2 = int(len(idx)*.70), int(len(idx)*.85)
        for dst, part in zip(subsets, (idx[:n1], idx[n1:n2], idx[n2:])):
            dst.extend(part)
    result = {}
    for name, idx in zip(('train', 'cal', 'val'), subsets):
        result['x_'+name] = (train[idx, :-1]/16).astype(np.float32)
        result['y_'+name] = train[idx, -1].astype(int)
    result['x_test'] = (test[:, :-1]/16).astype(np.float32)
    result['y_test'] = test[:, -1].astype(int)
    return result


def forward(m, x):
    h = np.maximum(x @ m['w1'] + m['b1'], 0)
    return h @ m['w2'] + m['b2']


def accuracy(m, x, y):
    return float(np.mean(forward(m, x).argmax(1) == y))


def train_model(data, epochs=50):
    """64 → 32 → 10，ReLU + 交叉熵；mini-batch SGD，非大模型训练。"""
    rng = np.random.default_rng(42)
    m = dict(w1=(rng.normal(size=(64,32))*np.sqrt(2/64)).astype('float32'),
             b1=np.zeros(32, 'float32'),
             w2=(rng.normal(size=(32,10))*np.sqrt(2/32)).astype('float32'),
             b2=np.zeros(10, 'float32'))
    x, y = data['x_train'], data['y_train']
    history = []
    for epoch in range(epochs):
        idx = rng.permutation(len(x))
        for start in range(0,len(x),128):
            xb, yb = x[idx[start:start+128]], y[idx[start:start+128]]
            h = np.maximum(xb@m['w1']+m['b1'],0)
            z = h@m['w2']+m['b2']; z -= z.max(1,keepdims=True)
            p = np.exp(z); p /= p.sum(1,keepdims=True)
            p[np.arange(len(yb)),yb] -= 1; p /= len(yb)
            dw2, db2 = h.T@p, p.sum(0)
            dh = (p@m['w2'].T)*(h>0)
            dw1, db1 = xb.T@dh, dh.sum(0)
            lr = .15
            for k,g in [('w1',dw1),('b1',db1),('w2',dw2),('b2',db2)]:
                m[k] -= lr*g
        history.append([epoch+1, accuracy(m,data['x_val'],data['y_val'])])
    return m, history


def quantize(w, bits, scale=None):
    """对称窄范围量化：每个输出通道一个 scale。W 形状为 [in,out]。"""
    if bits not in (2,4,8):
        raise ValueError('bits 必须为 2、4 或 8')
    qmax = 2**(bits-1)-1
    if scale is None:
        scale = np.maximum(np.max(np.abs(w), axis=0)/qmax, 1e-12)
    q = np.clip(np.rint(w/scale), -qmax, qmax).astype(np.int8)
    return (q*scale).astype(np.float32), q, np.asarray(scale,dtype=np.float32)


def gptq_educational(w, x, bits, damp=.01):
    """教学版逐通道补偿：固定尺度、无分组/重排/分块/kernel。
    当前剩余维度的逆 Hessian 为 A。固定第一个误差后，用其余维度补偿。
    这是 GPTQ/OBS 的核心二次目标；不是生产 GPTQ 实现。
    """
    wc = w.astype(np.float64).copy()
    _, _, scale = quantize(w, bits)
    h = x.astype(np.float64).T@x.astype(np.float64)/len(x)
    h += np.eye(len(h))*max(damp*np.mean(np.diag(h)),1e-8)
    a = np.linalg.inv(h)
    codes = np.zeros_like(w,dtype=np.int8)
    for j in range(len(w)):
        qj, codes[j], _ = quantize(wc[j:j+1],bits,scale)
        err = wc[j]-qj[0]
        if j+1 < len(w):
            wc[j+1:] -= np.outer(a[1:,0]/a[0,0],err)
            a = a[1:,1:]-np.outer(a[1:,0],a[0,1:])/a[0,0]
    return codes*scale,codes,scale


def awq_inspired(w, x, bits):
    """AWQ 启发的缩放搜索；没有原算法完整的 clipping / mapping。
    测试集不参与选择。S = diag(mean(abs(X)) ** alpha)。
    """
    stat = np.maximum(np.mean(np.abs(x),axis=0),1e-4)
    y = x@w; best = None
    for alpha in (0,.25,.5,.75,1):
        s = stat**alpha
        s /= np.sqrt(s.max()*s.min())
        deq, q, scale = quantize(w*s[:,None],bits)
        effective = deq/s[:,None]
        loss = float(np.mean((x@effective-y)**2))
        if best is None or loss < best[0]:
            best = (loss,effective,q,scale,s.astype('float32'),alpha)
    return best[1:]


def pack_codes(q,bits):
    """真实打包为 byte：2 bit 每字节 4 个，4 bit 每字节 2 个。"""
    codes = (q.astype(np.int16).ravel() + (2**(bits-1)-1)).astype(np.uint8)
    per_byte = 8//bits
    padded = np.pad(codes,(0,(-len(codes))%per_byte)).reshape(-1,per_byte)
    result = np.zeros(len(padded),np.uint8)
    for k in range(per_byte): result |= padded[:,k] << (k*bits)
    return result.tobytes()


def unpack_codes(raw,bits,shape):
    packed=np.frombuffer(raw,dtype=np.uint8); per_byte=8//bits
    cols=[(packed>>(k*bits))&((1<<bits)-1) for k in range(per_byte)]
    u=np.stack(cols,axis=1).ravel()[:int(np.prod(shape))].astype(np.int16)
    return (u-(2**(bits-1)-1)).astype(np.int8).reshape(shape)


def save_packed(path,m,codes,scales,bits,extra_scales=None):
    """简明教学格式；头部 + int 权重 + fp32 scale/bias，可回读验证。"""
    blobs=[];meta={'bits':bits,'items':[]}
    for name,q in codes.items():
        raw=pack_codes(q,bits);blobs.append(raw)
        meta['items'].append({'name':name,'kind':'packed','shape':list(q.shape),'bytes':len(raw)})
    arrays={**{k+'_scale':v for k,v in scales.items()},'b1':m['b1'],'b2':m['b2']}
    arrays.update({k+'_input_scale':v for k,v in (extra_scales or {}).items()})
    for name,v in arrays.items():
        raw=np.asarray(v,dtype='<f4').tobytes();blobs.append(raw)
        meta['items'].append({'name':name,'kind':'f32','shape':list(v.shape),'bytes':len(raw)})
    head=json.dumps(meta,separators=(',',':')).encode()
    path.write_bytes(len(head).to_bytes(4,'little')+head+b''.join(blobs))


def load_packed(path):
    raw=path.read_bytes();n=int.from_bytes(raw[:4],'little');meta=json.loads(raw[4:4+n]);pos=4+n;v={}
    for item in meta['items']:
        block=raw[pos:pos+item['bytes']];pos+=item['bytes']
        v[item['name']]=(unpack_codes(block,meta['bits'],item['shape']) if item['kind']=='packed'
                         else np.frombuffer(block,dtype='<f4').reshape(item['shape']).copy())
    m={'b1':v['b1'],'b2':v['b2']}
    for name in ('w1','w2'):
        m[name]=v[name]*v[name+'_scale']
        if name+'_input_scale' in v: m[name] /= v[name+'_input_scale'][:,None]
    return m


def gradient_descent():
    x=np.array([-1.,1.]);y=np.array([-1.,3.]);a=b=0.
    rows=[]
    for k in range(4):
        err=a*x+b-y;loss=np.mean(err**2)/2
        rows.append({'round':k,'a':a,'b':b,'loss':float(loss)})
        ga,gb=np.mean(err*x),np.mean(err)
        a,b=a-.8*ga,b-.8*gb  # 同时更新；不能先改 a 再算 b 的梯度
    return rows


def toy_compensation():
    x=np.array([[1.,1.],[2.,2.1],[3.,2.9]])
    w=np.array([.53,-.20]);baseline=x@w
    delta=float((x[:,1]@x[:,0])/(x[:,1]@x[:,1])*.03)
    rtn=np.array([.50,-.20]);comp=np.array([.50,-.20+delta])
    return {'delta':delta,'rtn_sse':float(np.sum((x@rtn-baseline)**2)),
            'compensated_sse':float(np.sum((x@comp-baseline)**2)),
            'note':'只固定第一个权重；第二个仍为浮点，演示一次补偿，不是最终全量化模型。'}


def evaluate(args):
    start=time.perf_counter();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    d=load_data();model_path=ROOT/'baseline.npz'
    if args.retrain or not model_path.exists():
        m,hist=train_model(d);np.savez(model_path,**m)
        (out/'training.json').write_text(json.dumps(hist))
    else:
        with np.load(model_path) as f:m={k:f[k] for k in f.files}
    count=min(args.calibration,len(d['x_cal']))
    if count<1: raise ValueError('校准样本数至少为 1')
    # 分层切分后混洗，避免前几个样本只来自数字 0。
    idx=np.random.default_rng(123).permutation(len(d['x_cal']))[:count]
    xc=d['x_cal'][idx]
    activations={'w1':xc,'w2':np.maximum(xc@m['w1']+m['b1'],0)}
    fp_logits=forward(m,d['x_test']);fpacc=accuracy(m,d['x_test'],d['y_test'])
    fpbytes=sum(v.nbytes for v in m.values())
    rows=[{'method':'FP32','bits':32,'accuracy':fpacc,'accuracy_drop_pp':0.,
           'weight_mse':0.,'test_logit_mse':0.,'packed_payload_bytes':fpbytes,
           'actual_file_bytes':model_path.stat().st_size,'calibration':0}]
    for bits in args.bits:
        for method in ('RTN','GPTQ-edu','AWQ-inspired'):
            qm={k:v.copy() for k,v in m.items()};codes={};scales={};extras={};alphas={}
            for name in ('w1','w2'):
                if method=='RTN': q,c,s=quantize(m[name],bits)
                elif method=='GPTQ-edu':q,c,s=gptq_educational(m[name],activations[name],bits)
                else:
                    q,c,s,extra,alpha=awq_inspired(m[name],activations[name],bits)
                    extras[name]=extra;alphas[name]=alpha
                qm[name]=q.astype('float32');codes[name]=c;scales[name]=s
            path=out/f'{method}_{bits}bit.qbin';save_packed(path,qm,codes,scales,bits,extras)
            restored=load_packed(path)
            for k in qm: np.testing.assert_allclose(restored[k],qm[k],rtol=2e-6,atol=2e-6)
            payload=sum(len(pack_codes(q,bits)) for q in codes.values())
            payload+=sum(v.nbytes for v in scales.values())+m['b1'].nbytes+m['b2'].nbytes
            payload+=sum(v.nbytes for v in extras.values())
            acc=accuracy(qm,d['x_test'],d['y_test'])
            row={'method':method,'bits':bits,'accuracy':acc,'accuracy_drop_pp':100*(fpacc-acc),
                 'weight_mse':float(np.mean(np.concatenate([(qm[k]-m[k]).ravel() for k in ('w1','w2')])**2)),
                 'test_logit_mse':float(np.mean((forward(qm,d['x_test'])-fp_logits)**2)),
                 'packed_payload_bytes':payload,'actual_file_bytes':path.stat().st_size,
                 'calibration':0 if method=='RTN' else count}
            rows.append(row)
            print(f'{method:13} {bits} bit  acc={acc:.4f}  logit_mse={row["test_logit_mse"]:.6f}  file={path.stat().st_size} B')
    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
          'seconds':time.perf_counter()-start,'split':{k:len(v) for k,v in d.items() if k.startswith('x_')},
          'seed':42,'calibration':count,'baseline_accuracy':fpacc,
          'gd':gradient_descent(),'toy':toy_compensation(),
          'limits':'推理先解包/反量化为 float32，再使用 NumPy 浮点矩阵乘。没有 INT4/INT8 加速 kernel；文件压缩不等于 RAM 减少或推理加速。'}
    (out/'run_info.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    make_report(out,rows,meta,d,m)
    print(f'FP32 accuracy={fpacc:.4f}; completed in {meta["seconds"]:.2f}s. Open {out/"report.html"}')
    return rows


def make_report(out,rows,meta,d,m):
    trs=''.join('<tr>'+''.join(f'<td>{r[k]:.5g}</td>' if isinstance(r[k],float) else f'<td>{r[k]}</td>' for k in r)+'</tr>' for r in rows)
    # HTML/SVG 数据图不依赖 matplotlib，断网可看。
    bars=''
    for j,r in enumerate(rows):
        yy=25+j*32;bars+=f'<text x="0" y="{yy+15}" font-size="12">{r["method"]} {r["bits"]}b</text><rect x="165" y="{yy}" width="{r["accuracy"]*480}" height="22" fill="#961318"/><text x="655" y="{yy+16}">{r["accuracy"]*100:.2f}%</text>'
    samples='';pred=forward(m,d['x_test'][:12]).argmax(1)
    for n,x in enumerate(d['x_test'][:12]):
        pixels=''.join(f'<rect x="{i%8*10}" y="{i//8*10}" width="10" height="10" fill="rgb({int(v*255)},{int(v*255)},{int(v*255)})"/>' for i,v in enumerate(x))
        samples+=f'<div><svg width="80" height="80">{pixels}</svg><br>标签 {d["y_test"][n]} / FP32 {pred[n]}</div>'
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CPU量化实训结果</title><style>body{{max-width:1250px;margin:35px auto;font:16px sans-serif;color:#222}}h1,h2{{color:#961318}}td,th{{padding:10px;border:1px solid #ddd}}table{{border-collapse:collapse;font-size:13px}}.digits{{display:flex;gap:18px;flex-wrap:wrap}}p{{line-height:1.8}}</style><h1>CPU 量化实训结果</h1><p>64 → 32 → 10 的小型手写数字分类器。FP32 基线准确率 {meta['baseline_accuracy']:.4%}。本次用时 {meta['seconds']:.2f} 秒。</p><p>{meta['limits']}</p><div class="digits">{samples}</div><h2>准确率对比</h2><svg width="800" height="{len(rows)*32+40}">{bars}</svg><h2>完整记录</h2><table><tr>{''.join('<th>'+k+'</th>' for k in rows[0])}</tr>{trs}</table><p>accuracy_drop_pp：相比 FP32 的准确率下降（百分点）；weight_mse：有效浮点权重的均方误差；test_logit_mse：量化模型与基线在测试集上的 logits 均方误差；packed_payload_bytes：权重编码、尺度及偏置的有效字节数；actual_file_bytes：包括文件头的实际文件大小。</p><p>GPTQ-edu 是固定尺度的逐通道二次误差补偿；AWQ-inspired 是激活统计驱动的缩放网格搜索。两者用于理解思想，不复现完整 LLM 算法，也不保证在每个位宽、每个样本集上优于 RTN。测试集只用于报告，不能用其结果选择超参数。</p></html>'''
    (out/'report.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bits',nargs='+',type=int,choices=[2,4,8],default=[8,4,2])
    p.add_argument('--calibration',type=int,default=128)
    p.add_argument('--out',default='results')
    p.add_argument('--retrain',action='store_true')
    args=p.parse_args();evaluate(args)
