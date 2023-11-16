import unittest

import numpy as np
import torch
from confpred.softsort.neural_sort import soft_quantile


class TestSoftQuantile(unittest.TestCase):
    def test_1d_array(self):
        a = torch.tensor([1., 3., 5., 7., 8.]).float()
        q = 0.5
        res = soft_quantile(a, q)
        self.assertAlmostEqual(res.item(), 5., places=1)

    def test_1d_array_mixed(self):
        a = torch.tensor([3., 5., 1., 8., 7.]).float()
        q = 0.5
        res = soft_quantile(a, q)
        self.assertAlmostEqual(res.item(), 5., places=1)

    def test_1d_array_multiq(self):
        a = torch.rand(100).float()
        q = [0.25,0.75]
        res = soft_quantile(a, q)
        res2 = torch.quantile(a, torch.tensor(q))
        self.assertAlmostEqual(res[0].item(), res2[0].item(), places=1)
        self.assertAlmostEqual(res[1].item(), res2[1].item(), places=1)

    def test_1d_array_multiq_rescaled(self):
        a = 0.01*torch.rand(100).float()
        q = [0.25,0.75]
        res = soft_quantile(a, q, tau=0.001)
        res2 = torch.quantile(a, torch.tensor(q))
        self.assertAlmostEqual(res[0].item(), res2[0].item(), places=2)
        self.assertAlmostEqual(res[1].item(), res2[1].item(), places=2)

    def test_2d_array(self):
        a = torch.tensor([[3., 5., 1., 8., 7.],[1., 5., 2., 6., 9.]]).float()
        q = 0.5
        res = soft_quantile(a, q, dim=1)
        res2 = torch.quantile(a, torch.tensor(q), dim=1)
        self.assertAlmostEqual(res[0].item(), res2[0].item(), places=1)
        self.assertAlmostEqual(res[1].item(), res2[1].item(), places=1)
    
    def test_random_array(self):
        a = torch.rand(100)
        q = 0.9
        res = soft_quantile(a, q)
        res2 = torch.quantile(a, q)
        self.assertAlmostEqual(res.item(), res2.item(), places=1)

if __name__ == "__main__":
    unittest.main()
