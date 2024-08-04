from __future__ import print_function
import argparse
from timeit import default_timer as timer
import os

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
from model import Net


def train(model, data_loader, optimizer, epoch, writer):
    print('===> Training mode')

    num_batches = len(data_loader)  # iteration per epoch. e.g: 469
    total_step = args.epochs * num_batches
    epoch_tot_acc = 0

    # Switch to train mode
    model.train()

    if args.cuda:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module

    start_time = timer()
    epoch_tot_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(data_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (epoch * num_batches) - num_batches

        labels = target
        target_one_hot = utils.one_hot_encode(target, length=args.num_classes)
        assert target_one_hot.size() == torch.Size([batch_size, 4])

        data, target = Variable(data), Variable(target_one_hot)

        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        # Train step - forward, backward and optimize
        optimizer.zero_grad()
        output = model(data)  # output from DigitCaps (out_digit_caps)
        loss, margin_loss, recon_loss = model.loss(data, output, target)
        loss.backward()
        optimizer.step()

        # Calculate accuracy for each step and average accuracy for each epoch
        acc = utils.accuracy(output, labels, args.cuda)
        epoch_tot_acc += acc
        epoch_tot_loss += loss.item()
        epoch_avg_acc = epoch_tot_acc / (batch_idx + 1)

        # TensorBoard logging
        # 1) Log the scalar values
        writer.add_scalar('train/total_loss', loss.data.item(), global_step)
        writer.add_scalar('train/margin_loss',
                          margin_loss.data.item(), global_step)
        if args.use_reconstruction_loss:
            writer.add_scalar('train/reconstruction_loss',
                              recon_loss.data.item(), global_step)
        writer.add_scalar('train/batch_accuracy', acc, global_step)
        writer.add_scalar('train/accuracy', epoch_avg_acc, global_step)

        # 2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, utils.to_np(value), global_step)
            writer.add_histogram(
                tag + '/grad', utils.to_np(value.grad), global_step)

        # Print losses
        if batch_idx % args.log_interval == 0:
            template = 'Epoch {}/{}, ' \
                'Step {}/{}: ' \
                '[Total loss: {:.6f},' \
                '\tMargin loss: {:.6f},' \
                '\tReconstruction loss: {:.6f},' \
                '\tBatch accuracy: {:.6f},' \
                '\tAccuracy: {:.6f}]'
            tqdm.write(template.format(
                epoch,
                args.epochs,
                global_step,
                total_step,
                loss.data.item(),
                margin_loss.data.item(),
                recon_loss.data.item() if args.use_reconstruction_loss else 0,
                acc,
                epoch_avg_acc))

    # Print time elapsed for an epoch
    end_time = timer()
    print('Time elapsed for epoch {}: {:.0f}s.'.format(
        epoch, end_time - start_time))
    avg_train_loss =epoch_tot_loss / num_batches
    avg_train_accuracy =  epoch_avg_acc
    return avg_train_accuracy, avg_train_loss

def test(model, data_loader, num_train_batches, epoch, writer):
    print('===> Evaluate mode')

    # Switch to evaluate mode
    model.eval()

    if args.cuda:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module

    loss = 0
    margin_loss = 0
    recon_loss = 0

    correct = 0

    num_batches = len(data_loader)

    global_step = epoch * num_train_batches + num_train_batches

    for data, target in data_loader:
        batch_size = data.size(0)
        target_indices = target
        target_one_hot = utils.one_hot_encode(
            target_indices, length=args.num_classes)
        assert target_one_hot.size() == torch.Size([batch_size, 4])

        data, target = Variable(data, volatile=True), Variable(target_one_hot)

        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        # Output predictions
        output = model(data)  # output from DigitCaps (out_digit_caps)

        # Sum up batch loss
        t_loss, m_loss, r_loss = model.loss(
            data, output, target, size_average=False)
        loss += t_loss.data[0].item()
        margin_loss += m_loss.data[0].item()
        recon_loss += r_loss.data[0].item()

        # Count number of correct predictions
        # v_magnitude shape: [128, 10, 1, 1]
        v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
        # pred shape: [128, 1, 1, 1]
        pred = v_magnitude.data.max(1, keepdim=True)[1].cpu()
        correct += pred.eq(target_indices.view_as(pred)).sum()

    # Get the reconstructed images of the last batch
    if args.use_reconstruction_loss:
        reconstruction = model.decoder(output, target)
        # Input image size and number of channel.
        # By default, for MNIST, the image width and height is 28x28 and 1 channel for black/white.
        image_width = args.input_width
        image_height = args.input_height
        image_channel = args.num_conv_in_channel
        recon_img = reconstruction.view(-1,
                                        image_channel, image_width, image_height)
        assert recon_img.size() == torch.Size(
            [batch_size, image_channel, image_width, image_height])

        # Save the image into file system
        utils.save_image(
            recon_img, 'results/recons_image_test_{}_{}.png'.format(epoch, global_step))
        utils.save_image(
            data, 'results/original_image_test_{}_{}.png'.format(epoch, global_step))

        # Add and visualize the image in TensorBoard
        recon_img = vutils.make_grid(
            recon_img.data, normalize=True, scale_each=True)
        original_img = vutils.make_grid(
            data.data, normalize=True, scale_each=True)
        writer.add_image('test/recons-image-{}-{}'.format(epoch,
                         global_step), recon_img, global_step)
        writer.add_image('test/original-image-{}-{}'.format(epoch,
                         global_step), original_img, global_step)

    # Log test losses
    loss /= num_batches
    margin_loss /= num_batches
    recon_loss /= num_batches

    # Log test accuracies
    num_test_data = len(data_loader.dataset)
    accuracy = correct / num_test_data
    accuracy_percentage = 100. * accuracy

    # TensorBoard logging
    # 1) Log the scalar values
    writer.add_scalar('test/total_loss', loss, global_step)
    writer.add_scalar('test/margin_loss', margin_loss, global_step)
    if args.use_reconstruction_loss:
        writer.add_scalar('test/reconstruction_loss', recon_loss, global_step)
    writer.add_scalar('test/accuracy', accuracy, global_step)

    # Print test losses and accuracy
    print('Test: [Loss: {:.6f},'
          '\tMargin loss: {:.6f},'
          '\tReconstruction loss: {:.6f}]'.format(
              loss,
              margin_loss,
              recon_loss if args.use_reconstruction_loss else 0))
    print('Test Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, num_test_data, accuracy_percentage))

    return accuracy, loss
    

def plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses):
    epochs = range(1, len(test_accuracies) + 1)
    
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.savefig('results/loss_plot.png')
    plt.close() 
    
    plt.figure()
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')
    plt.savefig('results/accuracy_plot.png')
    plt.close() 
    
    
def main():
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Example of Capsule Network')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs. default=10')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate. default=0.01')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size. default=128')
    parser.add_argument('--test-batch-size', type=int,
                        default=128, help='testing batch size. default=128')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status. default=10')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training. default=false')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')
    parser.add_argument('--num-conv-out-channel', type=int, default=256,
                        help='number of channels produced by the convolution. default=256')
    parser.add_argument('--num-conv-in-channel', type=int, default=1,
                        help='number of input channels to the convolution. default=1')
    parser.add_argument('--num-primary-unit', type=int, default=8,
                        help='number of primary unit. default=8')
    parser.add_argument('--primary-unit-size', type=int,
                        default=1152, help='primary unit size is 32 * 6 * 6. default=1152')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='number of digit classes. 1 unit for one MNIST digit. default=10')
    parser.add_argument('--output-unit-size', type=int,
                        default=16, help='output unit size. default=16')
    parser.add_argument('--num-routing', type=int,
                        default=3, help='number of routing iteration. default=3')
    parser.add_argument('--use-reconstruction-loss', type=utils.str2bool, nargs='?', default=True,
                        help='use an additional reconstruction loss. default=True')
    parser.add_argument('--regularization-scale', type=float, default=0.0005,
                        help='regularization coefficient for reconstruction loss. default=0.0005')
    parser.add_argument(
        '--dataset', help='the name of dataset (mnist, cifar10)', default='capsule')
    parser.add_argument('--input-width', type=int,
                        default=28, help='input image width to the convolution. default=28 for MNIST')
    parser.add_argument('--input-height', type=int,
                        default=28, help='input image height to the convolution. default=28 for MNIST')

    args = parser.parse_args()

    print(args)

    # Check GPU or CUDA is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Get reproducible results by manually seed the random number generator
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_loader, test_loader = utils.load_data(args)

    # Build Capsule Network
    print('===> Building model')
    model = Net(num_conv_in_channel=args.num_conv_in_channel,
                num_conv_out_channel=args.num_conv_out_channel,
                num_primary_unit=args.num_primary_unit,
                primary_unit_size=args.primary_unit_size,
                num_classes=args.num_classes,
                output_unit_size=args.output_unit_size,
                num_routing=args.num_routing,
                use_reconstruction_loss=args.use_reconstruction_loss,
                regularization_scale=args.regularization_scale,
                input_width=args.input_width,
                input_height=args.input_height,
                cuda_enabled=args.cuda)

    if args.cuda:
        print('Utilize GPUs for computation')
        print('Number of GPU available', torch.cuda.device_count())
        model.cuda()
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    # Print the model architecture and parameters
    print('Model architectures:\n{}\n'.format(model))

    print('Parameters and size:')
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, list(param.size())))
    num_params = sum([param.nelement() for param in model.parameters()])

    print('\nTotal number of parameters: {}\n'.format(
        num_params + (11520 if args.dataset == 'mnist' else 20480)))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Make model checkpoint directory
    if not os.path.exists('results/trained_model'):
        os.makedirs('results/trained_model')

    # Set the logger
    writer = SummaryWriter()

    # Train and test
    train_accuracies, train_losses = [], []
    test_accuracies, test_losses = [], []
    for epoch in range(1, args.epochs + 1):
        avg_train_accuracy, avg_train_loss = train(model, train_loader, optimizer, epoch, writer)
        avg_test_accuracy, avg_test_loss = test(model, test_loader, len(train_loader), epoch, writer)
        train_accuracies.append(avg_train_accuracy)
        train_losses.append(avg_train_loss)
        
        test_accuracies.append(avg_test_accuracy)
        test_losses.append(avg_test_loss)
        
        # Save model checkpoint
        utils.checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, epoch)

    writer.close()

    plot_metrics(train_accuracies, train_losses, test_accuracies, test_losses)
if __name__ == "__main__":
    main()
