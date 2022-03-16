#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest

from .test_effiecient_frontier import TestEfficientFrontierWithAMPL
from .test_effiecient_frontier_models import TestEfficientFrontierModels
from .test_discrete_allocation import TestDiscreteAllocationWithAMPL

if __name__ == "__main__":
    unittest.main()
