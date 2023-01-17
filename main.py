from pruner.GraSP import GraSP
import timm 
from datagen import load_cifar10

if __name__ == "__main__":
    model = timm.create_model("resnet10t", pretrained = False, in_chans = 3, num_classes = 10)
    train_data, test_data = load_cifar10(batch_size = 128)
    
    print("50% pruning")
    GraSP(model, 0.5, train_data, "cpu")
    
    print("97% pruning")
    GraSP(model, 0.97, train_data, "cpu")
    
    print("98% pruning")
    GraSP(model, 0.98, train_data, "cpu")