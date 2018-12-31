os.environ["CUDA_VISIBLE_DEVICES"] = '1'

with open('/home/kamer/notebooks/data/G9_data/action_data.pkl', 'rb') as f:
       X, y, z = cPickle.load(f)

X_transformer = QuantileTransformer(output_distribution='uniform')
X = X_transformer.fit_transform(X.reshape(-1, 128)).reshape(-1, 8, 128)

from sklearn.model_selection import GroupKFold
group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, z)

from sklearn.utils.class_weight import compute_class_weight


epochs = 50

all_preds = []
all_targets = []

for train_index, test_index in group_kfold.split(X, y, z):

    model = Arch2(in_channels = 8, out_channels = 6, gap_size = 128)
    model.to(torch.device('cuda'))

    optimizer = AdamW(params = model.parameters(), lr = 1e-4)#2e-4

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


    cw = torch.Tensor(compute_class_weight('balanced', np.unique(y_train), y_train)).cuda()
    criterion = nn.CrossEntropyLoss(reduction='elementwise_mean', weight=cw).cuda()

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train.astype(int) - 1)

    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test.astype(int) - 1)


    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 2048, shuffle = True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 2048, shuffle = True)

    for epoch in range(epochs):
        print('====== Epoch %i ======' % epoch)


        model.train(True)

        total = 0.0

        preds = []
        targets = []

        for i, (batch_X, batch_y) in enumerate(train_loader):

            batch_X = Variable(batch_X).float().cuda()
            batch_t = Variable(batch_y).long().cuda()

            output = model(batch_X)
            loss = criterion(output, batch_t)

            preds.extend(output.argmax(1).detach().cpu().numpy())

            targets.extend(batch_y)

            total += loss.detach().cpu().numpy()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        fone = f1_score(np.array(targets), np.array(preds), average = 'weighted')
        print('Train %f F1 %f' % (total/(i+1), fone))

        model.eval()
        total = 0.0

        for i, (batch_X, batch_y) in enumerate(test_loader):
            batch_X = Variable(batch_X).float().cuda()
            batch_t = Variable(batch_y).long().cuda()

            output = model(batch_X)
            loss = criterion(output, batch_t)

            preds.extend(output.argmax(1).detach().cpu().numpy())

            targets.extend(batch_y)

            total += loss.detach().cpu().numpy()

        fone = f1_score(np.array(targets), np.array(preds), average = 'weighted')
        print('Eval %f F1 %f' % (total/(i+1), fone))

    all_preds.extend(preds)
    all_targets.extend(targets)

fone = f1_score(np.array(all_targets), np.array(all_preds), average = 'weighted')
print(fone)