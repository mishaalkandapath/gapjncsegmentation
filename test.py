import numpy as np
import os
import argparse
from dataset import *
from models import *
from utilities_train import *
from utilities import *

parser = argparse.ArgumentParser(description="Get evaluation metrics for the model")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
parser.add_argument("--results_dir", type=str, required=True, help="Path to the results directory")
args = parser.parse_args()

model_path = args.model_path
data_dir = args.data_dir
results_dir = args.results_dir
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model, optimizer, epoch, loss, batch_size, lr, focal_loss_weights = load_checkpoint(model, optimizer, model_path)
model = model.eval()
print(f"Model loaded from {model_path}")

x_valid_dir = os.path.join(data_dir, "original", "valid")
y_valid_dir = os.path.join(data_dir, "ground_truth", "valid")
valid_dataset = SliceDataset(x_valid_dir, y_valid_dir)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
print(f"Validation dataset loaded from {x_valid_dir} and {y_valid_dir}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_accuracy = 0
total_precision = 0
total_recall = 0
total_iou = 0
total_tp = 0
total_fp = 0
total_tn = 0
total_fn = 0
# num_samples = 3
num_samples = len(valid_dataset)

for i in range(num_samples):
    # get prediction
    image, mask = valid_dataset[i] # (channels, depth, height, width)
    image, mask = image.to(DEVICE), mask.to(DEVICE)
    intermediate_pred, pred = model(image)
    pred = torch.argmax(pred[0], dim=0) # (depth, height, width)
    
    # calculate the metrics
    iou = get_iou(pred=pred, target=mask[1])
    accuracy = get_accuracy(pred=pred, target=mask[1])
    precision = get_precision(pred=pred, target=mask[1])
    recall = get_recall(pred=pred, target=mask[1])
    tp, fp, fn, tn = get_confusion_matrix(pred=pred, target=mask[1])
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    total_iou += iou
    total_tp += tp
    total_fp += fp
    total_tn += tn
    
    # save the results
    fig, ax = plt.subplots(3, 5, figsize=(15, 5), num=f"valid_{i}")
    visualize_3d_slice(image[0], ax[0], "Input")
    visualize_3d_slice(mask[0], ax[1], "Label")
    visualize_3d_slice(pred, ax[2], "Prediction")
    plt.savefig(f"{results_dir}/valid_{i}.png")
    plt.close("all")
    
    print(f"TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn}")
    print(f"Valid {i}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, iou={iou:.4f} | progress: {100*(i+1)/num_samples:.2f}%")
avg_accuracy = total_accuracy / num_samples
avg_precision = total_precision / num_samples
avg_recall = total_recall / num_samples
avg_iou = total_iou / num_samples
avg_tp = total_tp / num_samples
avg_fp = total_fp / num_samples
avg_tn = total_tn / num_samples
avg_fn = total_fn / num_samples
print(f"AVERAGE accuracy: {avg_accuracy:.4f}, precision: {avg_precision:.4f}, recall: {avg_recall:.4f}, iou: {avg_iou:.4f}")