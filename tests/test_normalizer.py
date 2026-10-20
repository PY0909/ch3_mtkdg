"""normalizer 单元测试：train-only 拟合、往返一致性、身份指纹。"""

import numpy as np
import pytest

from mtkdg.data.normalizer import TrainOnlyStandardScaler


def test_fit_transform_roundtrip():
    rng = np.random.RandomState(0)
    train = rng.normal(5.0, 2.0, size=(500, 4))
    scaler = TrainOnlyStandardScaler().fit(train)
    z = scaler.transform(train)
    assert z.shape == train.shape
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-3)
    back = scaler.inverse_transform(z)
    assert np.allclose(back, train, atol=1e-3)


def test_train_only_statistics():
    """统计量必须等于 train 数据上的 mean/std，不受后续 transform 数据影响。"""
    rng = np.random.RandomState(1)
    train = rng.normal(0.0, 1.0, size=(100, 3))
    test = rng.normal(100.0, 50.0, size=(100, 3))
    scaler = TrainOnlyStandardScaler().fit(train)
    # 用 test 数据 transform 不改变 scaler 统计量
    _ = scaler.transform(test)
    assert np.allclose(scaler.mean, train.mean(axis=0), atol=1e-8)
    assert np.allclose(scaler.std ** 2, train.var(axis=0), atol=1e-4)


def test_transform_before_fit_raises():
    scaler = TrainOnlyStandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((3, 3)))


def test_3d_input_supported():
    rng = np.random.RandomState(2)
    train = rng.normal(0, 1, size=(20, 10, 5))
    scaler = TrainOnlyStandardScaler().fit(train)
    z = scaler.transform(rng.normal(0, 1, size=(3, 10, 5)))
    assert z.shape == (3, 10, 5)
    assert np.isfinite(z).all()


def test_identity_depends_on_statistics():
    a = TrainOnlyStandardScaler().fit(np.random.RandomState(3).normal(0, 1, (50, 2)))
    b = TrainOnlyStandardScaler().fit(np.random.RandomState(4).normal(0, 1, (50, 2)))
    c = TrainOnlyStandardScaler().fit(np.random.RandomState(3).normal(0, 1, (50, 2)))
    assert a.identity() != b.identity()
    assert a.identity() == c.identity()


def test_state_dict_roundtrip():
    rng = np.random.RandomState(5)
    train = rng.normal(1.0, 3.0, size=(80, 6))
    scaler = TrainOnlyStandardScaler().fit(train)
    clone = TrainOnlyStandardScaler.from_state_dict(scaler.state_dict())
    x = rng.normal(0, 1, size=(7, 6))
    assert np.allclose(clone.transform(x), scaler.transform(x))
    assert clone.identity() == scaler.identity()
