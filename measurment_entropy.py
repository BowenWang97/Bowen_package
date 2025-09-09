import torch

class entropy_estimation():

    def __init__(self, samples):

        super(entropy_estimation, self).__init__()

        self.samples = samples

    def histogram_density(self, bins = 30):
        
        histogram = torch.histc(self.samples, bins=bins, min=self.samples.min(), max=self.samples.max())

        probality = histogram / histogram.sum()
        probality = probality[probality > 0]

        entropy = - (probality * torch.log(probality + 1e-8)).sum()

        return entropy