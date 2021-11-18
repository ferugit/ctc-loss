import torch

from ctc_loss import CTCLoss

def main():

    ctc = CTCLoss()
    ctc_torch = torch.nn.functional.ctc_loss

    num_labels = 3
    blank_index = num_labels-1 # last output = blank
    batch_size = 1

    T = torch.LongTensor([5])
    U = torch.LongTensor([2])
    y = torch.randint(low=0, high=num_labels-1, size=(U[0],)).unsqueeze(0).long()
    print('\nTarget: ')
    print('\t', y)
    print('\tShape: ', y.shape)

    x = torch.randn(max(T), batch_size, num_labels).log_softmax(2).detach().requires_grad_()
    print('\nEncoded: ')
    print('\t', x)
    print('\tShape: ', x.shape)


    # Calculate CTC loss
    loss = ctc(x, y , T, U, reduction='none', blank=blank_index)
    print(loss)

    # Use pytorch
    loss = ctc_torch(x, y , T, U, reduction='none', blank=blank_index)
    print(loss)


if __name__ == '__main__':
    main()