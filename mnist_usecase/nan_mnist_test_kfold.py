from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models.feature_extraction import create_feature_extractor
import os
import pickle
# from nan_ops import NaNLinear, NaNPool2d, count_skip_conv2d, NaNConv2d
from nan_ops import NaNPool2d, count_skip_conv2d, NaNConv2d
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import Subset

print( os.environ['THRESHOLD'] )
try:
    THRESH = float(os.environ['THRESHOLD'])
except:
    THRESH = os.environ['THRESHOLD']
EPSILON = float(os.environ['EPSILON'])
POOL_THRESH = float(os.environ['POOL_THRESH'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*11*11, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #x = self.conv1(x)
        x = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv1.weight, bias=self.conv1.bias, stride=1, padding=0)(x)
        #pickle.dump(x, open('/mnist/embeddings/conv1.pkl', 'wb'))
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x, _ = NaNPool2d(rtol_epsilon=EPSILON)(x, (2, 2), (2, 2))
        #pickle.dump(x, open('/mnist/embeddings/pool.pkl', 'wb'))
        #x = self.conv2(x)
        x = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=0)(x)
        #pickle.dump(x, open('/mnist/embeddings/conv2.pkl', 'wb'))
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #pickle.dump(x, open('/mnist/embeddings/flatten.pkl', 'wb'))
        #x = self.fc1(x)
        x = NaNLinear(threshold=THRESH, train=False, weight=self.fc1.weight, bias=self.fc1.bias)(x)
        #pickle.dump(x, open('/mnist/embeddings/fc1.pkl', 'wb'))
        pickle.dump(x, open('/mnist/embeddings/ieee_fc1.pkl', 'wb'))
        x = F.relu(x)
        x = self.dropout2(x)
        #x = self.fc2(x)
        x = NaNLinear(threshold=THRESH, train=False, weight=self.fc2.weight, bias=self.fc2.bias)(x)
        #pickle.dump(x, open('/mnist/embeddings/fc2.pkl', 'wb'))
        pickle.dump(x, open('/mnist/embeddings/ieee_fc2.pkl', 'wb'))
        output = F.log_softmax(x, dim=1)
        return output


class ConvOnlyNet(nn.Module):
    def __init__(self):
        super(ConvOnlyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 10, kernel_size=3, stride=1, padding=1)  # 10 channels for 10 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # pickle.dump(x, open(f'embeddings/conv1_{THRESH}_float16.pkl', 'wb'))

        x = F.max_pool2d(x, 2)  # Downsample
        # pickle.dump(x, open(f'embeddings/pool1_{THRESH}_float16.pkl', 'wb'))
        x = F.relu(self.conv2(x))
        # pickle.dump(x, open(f'embeddings/conv2_{THRESH}_float16.pkl', 'wb'))

        x = F.max_pool2d(x, 2)  # Downsample
        # pickle.dump(x, open(f'embeddings/pool2_{THRESH}_float16.pkl', 'wb'))
        x = F.relu(self.conv3(x))
        # pickle.dump(x, open(f'embeddings/conv3_{THRESH}_float16.pkl', 'wb'))

        x = F.max_pool2d(x, 2)  # Downsample
        pickle.dump(x, open(f'embeddings/pool3_{THRESH}_float16.pkl', 'wb'))
        x = self.conv4(x)  # No activation here, since this is the class feature map
        # pickle.dump(x, open(f'embeddings/conv4_{THRESH}_float16.pkl', 'wb'))
        
        # Global Average Pooling over the remaining spatial dimensions
        x = F.max_pool2d(x, (1, 1))
        # pickle.dump(x, open(f'embeddings/pool4_{THRESH}_float16.pkl', 'wb'))
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_classes)
        
        # Softmax for class probabilities
        return F.log_softmax(x, dim=1)

class NaNConvOnlyNet(nn.Module):
    def __init__(self):
        super(NaNConvOnlyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 10, kernel_size=3, stride=1, padding=1)  # 10 channels for 10 classes

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        conv1_skipped, conv1_total = count_skip_conv2d(x, self.conv1.weight.data, padding=1, stride=1, threshold=THRESH)
        print(f"Conv1: {conv1_skipped/conv1_total}")
        x = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv1.weight, bias=self.conv1.bias, stride=1, padding=1)(x)
        pickle.dump(x, open(f'embeddings/conv1_{THRESH}_float16.pkl', 'wb'))
        x = F.relu(x)

        # x = F.max_pool2d(x, 2)  # Downsample
        x, _ = NaNPool2d(max_threshold=POOL_THRESH, rtol_epsilon=EPSILON)(x, (2, 2), (2, 2))
        pickle.dump(x, open(f'embeddings/pool1_{THRESH}_float16.pkl', 'wb'))
        # x = F.relu(self.conv2(x))
        conv2_skipped, conv2_total = count_skip_conv2d(x, self.conv2.weight.data, padding=1, stride=1, threshold=THRESH)
        print(f"Conv2: {conv2_skipped/conv2_total} {conv2_skipped} {conv2_total}")
        x = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv2.weight, bias=self.conv2.bias, stride=1, padding=1)(x)
        pickle.dump(x, open(f'embeddings/conv2_{THRESH}_float16.pkl', 'wb'))
        x = F.relu(x)

        # x = F.max_pool2d(x, 2)  # Downsample
        x, _ = NaNPool2d(max_threshold=POOL_THRESH, rtol_epsilon=EPSILON)(x, (2, 2), (2, 2))
        pickle.dump(x, open(f'embeddings/pool2_{THRESH}_float16.pkl', 'wb'))
        # x = F.relu(self.conv3(x))
        conv3_skipped, conv3_total = count_skip_conv2d(x, self.conv3.weight.data, padding=1, stride=1, threshold=THRESH)
        print(f"Conv3: {conv3_skipped/conv3_total}")
        x = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv3.weight, bias=self.conv3.bias, stride=1, padding=1)(x)
        pickle.dump(x, open(f'embeddings/conv3_{THRESH}_float16.pkl', 'wb'))
        x = F.relu(x)

        # x = F.max_pool2d(x, 2)  # Downsample
        x, _ = NaNPool2d(max_threshold=POOL_THRESH, rtol_epsilon=EPSILON)(x, (2, 2), (2, 2))
        pickle.dump(x, open(f'embeddings/pool3_{THRESH}_float16.pkl', 'wb'))
        # x = self.conv4(x)  # No activation here, since this is the class feature map
        conv4_skipped, conv4_total = count_skip_conv2d(x, self.conv4.weight.data, padding=1, stride=1, threshold=THRESH)
        print(f"Conv4: {conv4_skipped/conv4_total}")
        x = NaNConv2d(train=False,threshold=THRESH, kernel=self.conv4.weight, bias=self.conv4.bias, stride=1, padding=1)(x)
        pickle.dump(x, open(f'embeddings/conv4_{THRESH}_float16.pkl', 'wb'))
        
        # Global Average Pooling over the remaining spatial dimensions
        # x = F.max_pool2d(x, (1, 1))
        x, _ = NaNPool2d(max_threshold=POOL_THRESH, rtol_epsilon=EPSILON)(x, (1, 1), (1, 1))
        pickle.dump(x, open(f'embeddings/pool4_{THRESH}_float16.pkl', 'wb'))
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_classes)
        
        # Softmax for class probabilities
        return F.log_softmax(x, dim=1)



def test(model, device, test_loader):
    # model.eval()
    test_loss = 0
    correct = 0
    metrics = {'pred': [], 'target': []}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            metrics['pred'].extend(pred.cpu().numpy())
            metrics['target'].extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return metrics, accuracy


def cross_validate(model, device, dataset, n_splits=5, batch_size=64):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Create data loaders for this fold
        val_subset = Subset(dataset, val_idx)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Test the model on the validation set
        metrics, accuracy = test(model, device, val_loader)
        all_metrics.append(metrics)
        fold_accuracies.append(accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    print(f"Cross-Validation Mean Accuracy: {mean_accuracy:.2f}%")
    return all_metrics, mean_accuracy


def cross_validate_reduced(model, device, dataset, n_splits=5, batch_size=64, precision=None):
    """
    Perform k-fold cross-validation using optional reduced precision.

    Args:
        model: The PyTorch model to evaluate.
        device: The device to run inference on.
        dataset: The dataset to split.
        n_splits (int): Number of CV folds.
        batch_size (int): Batch size for validation.
        precision (str or None): 'float16', 'bfloat16', or None.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []
    fold_accuracies = []

    if precision not in (None, 'float16', 'bfloat16'):
        raise ValueError("precision must be one of: None, 'float16', or 'bfloat16'")

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nFold {fold + 1}/{n_splits}")

        model.eval()

        val_subset = Subset(dataset, val_idx)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Use autocast for reduced precision inference
        if precision:
            dtype = dtype_map[precision]
            with torch.autocast(device_type=device.type, dtype=dtype):
                metrics, accuracy = test(model, device, val_loader)
        else:
            metrics, accuracy = test(model, device, val_loader)

        all_metrics.append(metrics)
        fold_accuracies.append(accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    print(f"\nCross-Validation Mean Accuracy: {mean_accuracy:.2f}%")
    return all_metrics, mean_accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For loading the model')
    parser.add_argument('--model-path', type=str)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=False, transform=transform)

    # model = ConvOnlyNet().to(device) #uninstrumented baseline model
    model = NaNConvOnlyNet().to(device) #instrumented NaN model
    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))

    all_metrics, mean_accuracy = cross_validate_reduced(model, device, dataset, n_splits=5, batch_size=args.test_batch_size, precision='bfloat16')
    pickle.dump(all_metrics, open(f"nan_test_metrics_{THRESH}_bfloat16.pkl", 'wb'))


if __name__ == '__main__':
    main()
