import copy
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler # <--- Added Import
from src.dataset import KolmogorovDataset_Rozet, TurbulenceDataset, KuramotoSivashinskyDataset
from src.data_transformations import DataParams, Transforms

def get_data_loaders(data_params, is_distributed=False):
    """
    Initializes and returns the data loaders for training and validation.
    Args:
        data_params: Dictionary containing dataset parameters.
        is_distributed: Boolean, whether to use DistributedDataParallel (DDP).
    """
    
    # --- Helper to create loaders with DDP logic ---
    def create_loader(dataset, batch_size, is_training, num_workers, drop_last=False):
        if is_distributed:
            # DDP: Use DistributedSampler
            # shuffle=True for training to randomize order per epoch
            # shuffle=False for val/test to keep order deterministic (though split across GPUs)
            sampler = DistributedSampler(dataset, shuffle=is_training, drop_last=drop_last)
            shuffle = False # DataLoader shuffle must be False if sampler is provided
        else:
            # Standard: Let DataLoader handle shuffling for training
            sampler = None
            shuffle = is_training 
            
            # Special case: If the original code explicitly used RandomSampler/SequentialSampler
            # outside of the DataLoader shuffle arg, we handle that in the specific blocks below.
            # But for the generic Kolmogorov/Kuramoto blocks, this default works.
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            sampler=sampler, 
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True # Recommended for GPU training
        )
        return loader

    if data_params["dataset_name"] == "KolmogorovFlow":
        
        trainSet = KolmogorovDataset_Rozet(
            "Test",
            data_params["data_path"],
            mode="train",
            resolution=data_params["resolution"],
            sequenceLength=data_params["sequence_length"],
            framesPerTimeStep=data_params["frames_per_time_step"],
            limit_trajectories=data_params["limit_trajectories_train"]
        )
        valSet = KolmogorovDataset_Rozet(
            "Test",
            data_params["data_path"],
            mode="valid",
            resolution=data_params["resolution"],
            limit_trajectories=data_params["limit_trajectories_val"],
            sequenceLength=data_params["sequence_length"],
            framesPerTimeStep=data_params["frames_per_time_step"]
        )
        trajSet = KolmogorovDataset_Rozet(
            "Test",
            data_params["data_path"],
            mode="valid",
            resolution=data_params["resolution"],
            limit_trajectories=data_params["limit_trajectories_val"],
            sequenceLength=data_params["trajectory_sequence_length"],
            framesPerTimeStep=data_params["frames_per_time_step"]
        )

        train_loader = create_loader(trainSet, data_params["batch_size"], is_training=True, num_workers=0)
        val_loader = create_loader(valSet, data_params["val_batch_size"], is_training=False, num_workers=0)
        traj_loader = create_loader(trajSet, data_params["batch_size"], is_training=False, num_workers=0)

    elif data_params["dataset_name"] == "KuramotoSivashinsky":
        
        trainSet = KuramotoSivashinskyDataset(
            "Test",
            data_params["data_path"],
            mode="train",
            resolution=data_params["resolution"],
            sequenceLength=data_params["sequence_length"],
            limit_trajectories=data_params["limit_trajectories_train"]
        )
        valSet = KuramotoSivashinskyDataset(
            "Test",
            data_params["data_path"],
            mode="valid",
            resolution=data_params["resolution"],
            limit_trajectories=data_params["limit_trajectories_val"],
            sequenceLength=data_params["sequence_length"],
        )
        trajSet = KuramotoSivashinskyDataset(
            "Test",
            data_params["data_path"],
            mode="valid",
            resolution=data_params["resolution"],
            limit_trajectories=data_params["limit_trajectories_val"],
            sequenceLength=data_params["trajectory_sequence_length"],
        )

        train_loader = create_loader(trainSet, data_params["batch_size"], is_training=True, num_workers=0)
        val_loader = create_loader(valSet, data_params["val_batch_size"], is_training=False, num_workers=0)
        traj_loader = create_loader(trajSet, data_params["batch_size"], is_training=False, num_workers=0)

    elif data_params["dataset_name"] == "TransonicFlow":
        
        trainSet = TurbulenceDataset("Training", [data_params["data_path"]], filterTop=['128_tra'], filterSim=[[0,1,2,14,15,16,17,18]], excludefilterSim=True, filterFrame=[(0,1000)],
                    sequenceLength=[data_params["sequence_length"]], randSeqOffset=True, simFields=["dens", "pres"], simParams=["mach"], printLevel="sim")

        valSet = TurbulenceDataset("Training", [data_params["data_path"]], filterTop=['128_tra'], filterSim=[(16,19)], excludefilterSim=True,
                        filterFrame=[(500,750)], sequenceLength=[data_params["sequence_length"]], randSeqOffset=False, simFields=["dens", "pres"], simParams=["mach"], printLevel="sim")

        testSet = TurbulenceDataset("Test Extrapolate Mach 0.50-0.52", [data_params["data_path"]], filterTop=["128_tra"], filterSim=[(0,3)],
                            filterFrame=[(500,750)], sequenceLength=[[60,2]], randSeqOffset=False, simFields=["dens", "pres"], simParams=["mach"], printLevel="sim")

        p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=trainSet.sequenceLength, randSeqOffset=True,
                 dataSize=[128,64], dimension=2, simFields=["dens","pres"], simParams=["mach"], normalizeMode="machMixed")

        p_d_test = copy.deepcopy(p_d)
        p_d_test.augmentations = ["normalize"]
        p_d_test.sequenceLength = testSet.sequenceLength
        p_d_test.randSeqOffset = False
        p_d_test.batch = 4

        ### TRANSFORMS
        transTrain = Transforms(p_d)
        trainSet.transform = transTrain
        trainSet.printDatasetInfo()
        
        # --- Logic for TransonicFlow Samplers ---
        if is_distributed:
            # In DDP, DistributedSampler handles the shuffling/subsetting
            trainSampler = DistributedSampler(trainSet, shuffle=True, drop_last=True)
            valSampler = DistributedSampler(valSet, shuffle=False, drop_last=True)
            testSampler = DistributedSampler(testSet, shuffle=False, drop_last=False)
        else:
            # Original Logic
            trainSampler = RandomSampler(trainSet)
            valSampler = RandomSampler(valSet)
            testSampler = SequentialSampler(testSet)

        # Note: When using a custom sampler (Distributed or Random), 
        # the shuffle arg in DataLoader must be False.
        
        transTest = Transforms(p_d_test)
        valSet.transform = transTrain # Using train transform for val as in original
        testSet.transform = transTest
        testSet.printDatasetInfo()

        train_loader = DataLoader(trainSet, sampler=trainSampler, shuffle=False,
                    batch_size=p_d.batch, drop_last=True, num_workers=4, pin_memory=True)
        
        val_loader = DataLoader(valSet, sampler=valSampler, shuffle=False,
                    batch_size=p_d.batch, drop_last=True, num_workers=4, pin_memory=True)
        
        traj_loader = DataLoader(testSet, sampler=testSampler, shuffle=False,
                        batch_size=p_d_test.batch, drop_last=False, num_workers=4, pin_memory=True)
        
    return train_loader, val_loader, traj_loader