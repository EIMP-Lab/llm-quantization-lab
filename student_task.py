"""任务 A：补全三处 TODO 后运行 python student_task.py。
输入 W 的形状为 [输入通道, 输出通道]，每个输出通道独立一个尺度。
2 bit 采用 {-1,0,1}；4 bit 采用 {-7,...,7}；8 bit 采用 {-127,...,127}。
"""
import numpy as np


def symmetric_quantize(w, bits):
    qmax = 2 ** (bits - 1) - 1
    # TODO 1：对 axis=0 求最大绝对值，再除以 qmax；下界用 1e-12 避免除零。
    scale = None
    # TODO 2：w / scale，np.rint 取最近整数，np.clip 截断，转 np.int8。
    q = None
    # TODO 3：反量化，恢复为浮点近似。
    w_hat = None
    return q, scale, w_hat


if __name__ == '__main__':
    w = np.array([[.53, 0.], [-.20, 0.], [.71, 0.]], dtype=np.float32)
    q, scale, w_hat = symmetric_quantize(w, 4)
    if q is None or scale is None or w_hat is None:
        raise SystemExit('请先补全 symmetric_quantize 中三处 TODO，再运行检查。')
    assert q.dtype == np.int8
    assert np.isfinite(w_hat).all(), '全零通道必须安全处理'
    assert (np.abs(q) <= 7).all()
    np.testing.assert_allclose(scale[0], .71 / 7, rtol=1e-6)
    np.testing.assert_array_equal(q[:, 0], [5, -2, 7])
    np.testing.assert_allclose(w_hat, q * scale, rtol=1e-6)
    print('检查通过！\n整数编码：\n', q, '\n尺度：', scale, '\n反量化：\n', w_hat)
