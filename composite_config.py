class CompositeConfiguration:
    
    class Averages:
        def __init__(self, num_classes):
            self.avg = np.zeros(num_classes)
            

    def __init__(self, classes_list):    
        self.classes_list = classes_list
    
    def verify_dataset_compatibility(self, dataset):
 
        assert len(dataset.composite_labels) == len(self.classes_list), \
                "Lengths don't match: %d != %d" % (
                len(self.classes_list), len(dataset.composite_labels))
        
        for class_config, class_dataset in zip(self.classes_list, dataset.composite_labels):    
            if class_config != class_dataset:
                raise Exception("Class name mismatch: %s != %s" % (class_config, class_dataset))

    def uniform_reweighting(self, dataset):
        
        self.verify_dataset_compatibility(dataset)
        

        for data in dataset:
