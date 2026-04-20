:::writing block

Self-Pruning Neural Network on CIFAR-10
A PyTorch-based implementation of a self-pruning neural network that dynamically removes less important weights during training to improve efficiency while maintaining competitive accuracy on the CIFAR-10 dataset.

Overview
Traditional neural networks are often overparameterized, leading to unnecessary computational cost and memory usage. This project explores a self-pruning mechanism where the network learns to identify and eliminate redundant connections during training.

The goal is to:

Reduce model complexity

Maintain strong classification performance

Improve inference efficiency

Key Features
Dynamic weight pruning during training

Fully automated data handling pipeline (no manual dataset setup required)

Efficient GPU training with optimized DataLoader configuration

Modular PyTorch implementation for easy experimentation

Clean and reproducible training workflow

Project Structure
text
├── self_pruning_nn.py     # Main training + pruning script
├── model.py               # Model architecture (SelfPruningNet)
├── utils.py               # Helper functions (if applicable)
├── requirements.txt       # Dependencies
├── .gitignore             # Excludes data/ and other artifacts
└── README.md              # Project documentation
Installation
Clone the repository:

bash
git clone https://github.com/your-username/self-pruning-nn.git
cd self-pruning-nn
Install dependencies:

bash
pip install -r requirements.txt
How to Run
Simply execute the main script:

bash
python self_pruning_nn.py
The dataset will be downloaded automatically (first run only)

Training will start immediately

Model pruning will occur during training

Model Architecture
The core model, SelfPruningNet, is a fully connected neural network that operates on flattened CIFAR-10 images.

Input layer: 
3072
3072 features (32×32×3)

Hidden layers: Dense layers with activation functions (e.g., ReLU)

Output layer: 10 classes

Integrated pruning mechanism:

Weights below a learned or threshold value are progressively zeroed out

Encourages sparsity during training

Pruning Strategy
The network applies on-the-fly pruning during training:

Low-magnitude weights are identified as less important

These weights are either:

Masked (set to zero), or

Permanently removed depending on implementation

Pruning may occur:

At fixed intervals, or

Continuously during backpropagation

This results in a sparser and more efficient model without requiring post-training compression.

Data & Preprocessing Pipeline
This project is designed to be fully self-contained with respect to data handling. Users cloning the repository do not need to manually download or manage the CIFAR-10 dataset.

Automated Dataset Download
The dataset is automatically downloaded during the first execution of the training script.

This is handled via:

python
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
The dataset is stored locally in the ./data directory, which is excluded from version control using .gitignore.

Subsequent runs reuse the cached dataset.

Dataset Overview
60,000 color images (32×32 resolution)

10 classes

50,000 training images, 10,000 testing images

Data Augmentation & Preprocessing
Training transformations:

RandomHorizontalFlip()

RandomCrop(32, padding=4)

Common processing:

Conversion to tensors

Normalization:

python
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]
Images are flattened into 
3072
3072-dimensional vectors before entering the network.

Efficient Data Loading
python
DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
Batch size: 256

Multiprocessing: num_workers=2

Faster GPU transfer: pin_memory=True

Results
Typical outcomes you can report (update with your actual results):

Test Accuracy: ~XX%

Parameter Reduction: ~XX% after pruning

Training Time: ~X hours on GPU

You can also include:

Accuracy vs pruning curve

Sparsity progression plots

Future Improvements
Implement structured pruning (channel/filter pruning)

Add support for convolutional architectures

Integrate pruning schedules (gradual pruning)

Deploy model using TorchScript or ONNX

Compare with baseline (non-pruned model)

License
This project is open-source and available under the MIT License.

Acknowledgements
PyTorch and Torchvision

CIFAR-10 dataset creators

Research work on neural network pruning and sparsity

:::
