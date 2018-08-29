
from prody import Gamma

class ContactMapAndDistanceGamma(Gamma):


    def __init__(self, contact_map, cutoff, alpha):
        self.contact_map = contact_map
        self.cutoff = cutoff
        self.alpha = alpha

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        contact_prob = self.contact_map[i, j]

        cutoff_squared = self.cutoff * self.cutoff
        s = 1 - dist2 / cutoff_squared

        if contact_prob == -1.0:
            return s * s

        return self.alpha * contact_prob * contact_prob + (1 - self.alpha) * s * s


class SquaredDistanceGamma(Gamma):

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        cutoff_squared = self.cutoff * self.cutoff
        s = 1 - dist2 / cutoff_squared
        return s * s
