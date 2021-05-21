import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import torch_geometric.nn as hnn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GAT_Net_Class(torch.nn.Module):
    def __init__(self):
        super(GAT_Net_Class, self).__init__()
        self.conv1 = hnn.GATConv(datasetCora.num_features, 8, heads=8, 
                                 dropout=0.6,concat=True, 
                                 graph_weight = graph_weight)
        # self.conv1 = hnn.GATConv(datasetCora.num_features, 8, heads=8, dropout=0.6,concat=True)        
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = hnn.GATConv(8*8, datasetCora.num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x, attention_weight = self.conv1(x, data.edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    
    # seed_value = 10000009

    # loading dataset
    dataset_name = 'Cora' #Citeseer or Pubmed
    datasetCora = Planetoid(root=dataset_name, name=dataset_name)
    data = datasetCora[0]

    # بدست آوردن ویژگی‌های راس‌ها و مقایسه آن‌ها با دیگر راس‌های همسایه به منظور بدست آوردن وزن یال‌ها
#    graph_weight = torch.zeros((1,13264), dtype=torch.float32)
    # graph_weight = torch.zeros((1,13264), dtype=torch.float32)
    # graph_weight[0,:] = (torch.load('graph_weight.pt').to(torch.float32))
    samefeatures = torch.zeros(10556, dtype=torch.float)
    for i in range(10556):
        feat_vi = data.x[data.edge_index[0,i]]
        feat_vj = data.x[data.edge_index[1,i]]
        for j in range(1433):
            if feat_vi[j] == feat_vj[j] == 1:
                samefeatures[i] += 1
    self_feature_v = torch.zeros(2708)
    for i in range(2708):
        self_feature_v[i] = sum(data.x[i])
    graph_weight = torch((samefeatures, self_feature_v),1)
    # -------------------------------------------------------------------------------------------

    def train_model():
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


    def test_model():
        model.eval()
        logits, accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs



    number_of_epochs = 100
    number_of_trials = 20
    
    plotvalues = np.zeros((number_of_epochs, 4))

    for trial in range(0, number_of_trials):
        # find the best accuracies during execution
        train_best, test_best, val_best, epoch_best = 0, 0, 0, 0

        model = None
        model = GAT_Net_Class()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        # print("Processing...")

        for epoch in range(1, number_of_epochs):
            train_model()
            train_, val_, test_ = test_model()

            if (test_ > test_best):
                epoch_best = epoch

            train_best = max(train_, train_best)
            test_best = max(test_, test_best)
            val_best = max(val_, val_best)
            
            plotvalues[epoch] = [epoch, train_, val_, test_]

            log = 'Trial:{} --> Epoch: {:03d} --> accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(trial, epoch, train_, val_, test_))
        print("Trial {}-->Best accuracies: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
              .format(trial, train_best, val_best, test_best))
        
        
        # ----------------------------- Plot Code ----------------------------------
        plt.figure(dpi=300)
        plt.rc('ytick', labelsize=6)
        plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
        plt.yticks(np.arange(0, 1, step=0.02))
        line1 = plt.plot(plotvalues[:, 0], plotvalues[:, 1], 'g-', label='Train acc')
        line1 = plt.plot(plotvalues[:, 0], plotvalues[:, 2], 'r-', label='Validation acc')
        line1 = plt.plot(plotvalues[:, 0], plotvalues[:, 3], label='Test acc')

        # line1.set
        plt.title('Accuracies')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        text = "epoch={}, accuracy=%{:.3f}".format(epoch_best, (test_best * 100))
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="bottom")
        plt.annotate(text, xy=(epoch_best, test_best), xytext=(0.94, 0.96), **kw)

        plt.grid()
        # plt.savefig('./plots/plots_in_range/_flexible_weight_decay/Trial_{}_best_test_{}_hyperedge_weight {:.4f}.png'
        #             .format(trial, test_best, hyperedge_weight[0]), dpi=300)
        plt.show()
        # ------------------------------------------------------------------------


