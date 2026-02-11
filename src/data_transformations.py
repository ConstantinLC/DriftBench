import torch
import torch.nn.functional as F
import numpy as np

# OBTAINED FROM 
#
# @inproceedings{shehata2025improved,
#   title={Improved Sampling Of Diffusion Models In Fluid Dynamics With Tweedie's Formula},
#   author={Shehata, Youssef and Holzschuh, Benjamin and Thuerey, Nils},
#   booktitle={The Thirteenth International Conference on Learning Representations},
#   year={2025}
# }

class DataParams(object):
    def __init__(self, batch=4, augmentations=[], sequenceLength=[1,1], randSeqOffset=False,
                dataSize=[128,64], dimension=2, simFields=[], simParams=[], normalizeMode=""):
        self.batch          = batch             # batch size
        self.augmentations  = augmentations     # used data augmentations
        self.sequenceLength = sequenceLength    # number of simulation frames in one sequence
        self.randSeqOffset  = randSeqOffset     # randomize sequence starting frame
        self.dataSize       = dataSize          # target data size for scale/crop/cropRandom transformation
        self.dimension      = dimension         # number of data dimension
        self.simFields      = simFields         # which simulation fields are added (vel is always used) from ["dens", "pres"]
        self.simParams      = simParams         # which simulation parameters are added from ["rey", "mach"]
        self.normalizeMode  = normalizeMode     # which mean and std values from different data sets are used in normalization transformation

    @classmethod
    def fromDict(cls, d:dict):
        p = cls()
        p.batch          = d.get("batch",            -1)
        p.augmentations  = d.get("augmentations",    [])
        p.sequenceLength = d.get("sequenceLength",   [])
        p.randSeqOffset  = d.get("randSeqOffset",    False)
        p.dataSize       = d.get("dataSize",         -1)
        p.dimension      = d.get("dimension",        -1)
        p.simFields      = d.get("simFields",        [])
        p.simParams      = d.get("simParams",        [])
        p.normalizeMode  = d.get("normalizeMode",    "")
        return p

    def asDict(self) -> dict:
        return {
            "batch"          : self.batch,
            "augmentations"  : self.augmentations,
            "sequenceLength" : self.sequenceLength,
            "randSeqOffset"  : self.randSeqOffset,
            "dataSize"       : self.dataSize,
            "dimension"      : self.dimension,
            "simFields"      : self.simFields,
            "simParams"      : self.simParams,
            "normalizeMode"  : self.normalizeMode,
        }

class Transforms(object):
    p_d: DataParams

    def __init__(self, p_d:DataParams):

        assert all(aug in ["normalize", "flip", "crop", "resize"]
                        for aug in p_d.augmentations), "Invalid augmentation provided!"
        assert not ("crop" in p_d.augmentations and "resize" in p_d.augmentations
                        ), "Crop and resize augmentation not allowed at the same time!"
        assert (p_d.normalizeMode != ""), "Invalid normalization mode!"

        self.p_d = p_d
        self.normalize = "normalize" in p_d.augmentations
        self.flip = "flip" in p_d.augmentations
        self.crop = "crop" in p_d.augmentations
        self.resize = "resize" in p_d.augmentations
        self.outputSize = p_d.dataSize
        self.dim = p_d.dimension
        self.simFields = p_d.simFields
        self.simParams = p_d.simParams

        # mean and std statistics from whole dataset for normalization
        if self.dim == 2:
            l = self.p_d.normalizeMode.lower()
            if ("inc" in l and "mixed" in l) or ("karman" in l and "mixed" in l):
                # ORDER (fields): velocity (x,y), --, pressure, ORDER (params): rey, --, --
                self.normMean = np.array([0.444969, 0.000299, 0, 0.000586, 550.000000, 0, 0], dtype=np.float32)
                self.normStd =  np.array([0.206128, 0.206128, 1, 0.003942, 262.678467, 1, 1], dtype=np.float32)

            if ("tra" in l and "mixed" in l) or ("mach" in l and "mixed" in l):
                # ORDER (fields): velocity (x,y), density, pressure, ORDER (params): rey, mach, --
                self.normMean = np.array([0.560642, -0.000129, 0.903352, 0.637941, 10000.000000, 0.700000, 0], dtype=np.float32)
                self.normStd =  np.array([0.216987, 0.216987, 0.145391, 0.119944, 1, 0.118322, 1], dtype=np.float32)

            if "iso" in l and "single" in l:
                # ORDER (fields): velocity (x,y,z), pressure, ORDER (params): --, --, --
                self.normMean = np.array([-0.054618, -0.385225, -0.255757, 0.033446, 0, 0, 0], dtype=np.float32)
                self.normStd =  np.array([0.539194, 0.710318, 0.510352, 0.258235, 1, 1, 1], dtype=np.float32)

        # seeding once for single thread data loading
        self.randGen = np.random.RandomState(torch.random.initial_seed() % 4294967295)


    def __call__(self, sample:dict):
        # seeding in every call for multi thread data loading
        if torch.utils.data.get_worker_info():
            self.randGen = np.random.RandomState(torch.utils.data.get_worker_info().seed % 4294967295)

        data = sample["data"]
        simParameters = sample["simParameters"]
        allParameters = sample["allParameters"]
        obsMask = sample.get("obsMask", None)
        path = sample["path"]

        # normalization to std. normal distr. with zero mean and unit std via statistics from whole dataset
        # ORDER (fields): velocity (x,y), velocity z / density, pressure, ORDER (params): rey, mach, zslice
        if self.normalize:
            filterList = [0, 1] if self.dim == 2 else [0, 1, 2]
            if "dens" in self.simFields or "velZ" in self.simFields:
                filterList += [2] if self.dim == 2 else [3]
            if "pres" in self.simFields:
                filterList += [3] if self.dim == 2 else [4]
            if "rey" in self.simParams:
                filterList += [4] if self.dim == 2 else [5]
            if "mach" in self.simParams:
                filterList += [5] if self.dim == 2 else [6]
            if "zslice" in self.simParams:
                filterList += [6] if self.dim == 2 else [7]
            filterArr = np.array(filterList)
            filterArrParam = filterArr[-len(self.simParams):]

            if self.simParams:
                meanParam = self.normMean[filterArrParam].reshape((1,-1))
                stdParam = self.normStd[filterArrParam].reshape((1,-1))
                simParameters = (simParameters - meanParam) / stdParam

            meanData = self.normMean[filterArr].reshape((1,-1,1,1)) if self.dim == 2 else self.normMean[filterArr].reshape((1,-1,1,1,1))
            stdData = self.normStd[filterArr].reshape((1,-1,1,1)) if self.dim == 2 else self.normStd[filterArr].reshape((1,-1,1,1,1))
            if self.dim == 2:
                data = (data - meanData) / stdData
            elif self.dim == 3:
                data = (data - meanData) / stdData

        # random flip
        if self.flip:
            if self.dim == 2:
                rand = self.randGen.rand(2) > 0.5
                flipped = False
                if rand[0]:
                    data = np.flip(data, axis=2)
                    flipped = True
                if rand[1]:
                    data = np.flip(data, axis=3)
                    flipped = True
                if flipped:
                    data = data.copy() #prevent negative strides that has issues with torch tensor creation
            if self.dim == 3:
                raise NotImplementedError("Flip augmentation not supported for 3D yet!")

        # random crop
        if self.crop:
            if self.dim == 2:
                s = self.outputSize
                if data.shape[2] > s[0] or data.shape[3] > s[1]:
                    c1 = self.randGen.randint(0, data.shape[2] - s[0]+1)
                    c2 = self.randGen.randint(0, data.shape[3] - s[1]+1)
                    data = data[..., c1:c1+s[0], c2:c2+s[1]]
            if self.dim == 3:
                raise NotImplementedError("Crop augmentation not supported for 3D yet!")

        # toTensor
        result = torch.from_numpy(data)
        if obsMask is not None:
            obsMask = torch.from_numpy(obsMask)

        # resize
        if self.resize:
            result = F.interpolate(result, self.outputSize, mode="bilinear", align_corners=True)
            if obsMask is not None:
                obsMask = F.interpolate(obsMask, self.outputSize, mode="nearest", align_corners=True)

        outDict = {"data": result, "simParameters": simParameters, "allParameters": allParameters, "path": path}
        if obsMask is not None:
            outDict["obsMask"] = obsMask
        return outDict