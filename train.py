import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model import CNNLSTM
from data_loader.py import load_data

def train_model():
    xdata, label, subIdx = load_data()
    samplenum = label.shape[0]
    selectedchan = [28]
    xdata = xdata[:, selectedchan, :]
    channelnum = len(selectedchan)
    lr = 1e-2
    batch_size = 50
    n_epoch = 15
    sf = 128
    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = label[i]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_losses = []
    epoch_accuracies = []
    test_losses = []
    test_accuracies = []
    for i in range(1, 12):
        trainindx = np.where(subIdx != i)[0]
        xtrain = xdata[trainindx]
        x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, 3 * sf)
        y_train = ydata[trainindx]
        testindx = np.where(subIdx == i)[0]
        xtest = xdata[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, 3 * sf)
        y_test = ydata[testindx]
        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        my_net = CNNLSTM().double().to(device)
        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss().to(device)
        subject_losses = []
        subject_accuracies = []
        for epoch in range(n_epoch):
            total_loss = 0.0
            my_net.train()
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data
                input_data = inputs.to(device).double()
                class_label = labels.to(device).long()
                my_net.zero_grad()
                class_output = my_net(input_data)
                err_s_label = loss_class(class_output, class_label)
                err_s_label.backward()
                optimizer.step()
                total_loss += err_s_label.item()
            avg_loss = total_loss / len(train_loader)
            subject_losses.append(avg_loss)
            print(f"Subject {i} | Epoch [{epoch+1}/{n_epoch}] | Loss: {avg_loss:.4f}")
            my_net.eval()
            with torch.no_grad():
                x_test_tensor = torch.from_numpy(x_test).to(device).double()
                answer = my_net(x_test_tensor)
                probs = answer.cpu().numpy()
                preds = probs.argmax(axis=-1)
                acc = accuracy_score(y_test, preds)
                subject_accuracies.append(acc)
                test_loss = loss_class(answer, torch.tensor(y_test).to(device).long()).item()
                test_losses.append(test_loss)
                test_accuracies.append(acc)
            print(f"Subject {i} | Epoch [{epoch+1}/{n_epoch}] | Accuracy: {acc:.4f}")
        epoch_losses.append(subject_losses)
        epoch_accuracies.append(subject_accuracies)
    avg_accuracies = np.mean(np.array(epoch_accuracies), axis=0)
    plt.figure()
    plt.plot(range(1, n_epoch + 1), avg_accuracies, marker='o')
    plt.title('Average Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    plt.show()
    torch.save(my_net.state_dict(), "cnn_lstm_model.pth")