import torch
import torch.nn as nn

class genetic_algprithm():

    def __init__(self, boundary_low, boundary_high):

        super(genetic_algprithm, self).__init__()

        self.boundary_low = boundary_low
        self.boundary_high = boundary_high

    