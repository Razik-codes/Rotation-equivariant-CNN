import numpy as np
import pytest
from src.analysis.statistics import seed_summary, paired_seed_summary, clustered_bootstrap_difference


def test_student_t_known_answer_and_ddof():
    result=seed_summary([1,2,3])
    assert result['mean'] == 2
    assert result['sd'] == 1
    assert result['ci_high'] == pytest.approx(2+4.302652729911275/np.sqrt(3))
    assert seed_summary([.5])['ci_low'] is None
    assert seed_summary([1,1,1])['ci_low'] == 1


def test_paired_raw_values():
    result=paired_seed_summary({42:1,123:2,456:3},{42:0,123:1,456:2})
    assert result['mean'] == 1 and result['sd'] == 0
    with pytest.raises(ValueError):
        paired_seed_summary({42:1},{123:1})


def test_cluster_bootstrap_preserves_rotated_copies():
    a=np.array([0,1,0,1,1,0],float)
    b=np.zeros(6)
    one=clustered_bootstrap_difference(a,b,seed=5,repetitions=100)
    repeated=clustered_bootstrap_difference(np.tile(a,(3,4,1)),np.tile(b,(3,4,1)),seed=5,repetitions=100)
    assert one == repeated
    zero=clustered_bootstrap_difference(a,a,repetitions=100)
    assert zero['ci_low'] == zero['ci_high'] == 0
    grouped=clustered_bootstrap_difference(a,b,cluster_ids=[0,0,0,1,1,1],repetitions=100)
    assert grouped['clusters'] == 2
    assert grouped['mean'] == .5
