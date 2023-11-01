import torch
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from model import TransformerClassifier
from dataset import PatientDataset
from trainer import Trainer
import pickle
import os

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()

    # transformer parameters
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=2)

    # num_patients - 10 for track 1, 9 for track 2
    parser.add_argument('--num_patients', type=int, default=2)

    # input paths
    parser.add_argument('--features_path', type=str, help='features to use')
    parser.add_argument('--dataset_path', type=str, help='features to use') # to get relapse labels



    # learning params    
    parser.add_argument('--optimizer', type=str, help='optimizer (SGD, Adam)', choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--weight_decay', type=float, help='L2 regularization weight', default=5e-4)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=10)

    # checkpoint
    parser.add_argument('--save_path', type=str, help='path to save model checkpoints', default='checkpoints')


    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cuda')

    args = parser.parse_args()
    args.seq_len = args.window_size
            
    return args

def main():

    # parse arguments
    args = parse()

    
    device = args.device
    print('Using device', args.device)

    # Model
    models = [TransformerClassifier(vars(args)).to(device) for _ in range(args.num_patients)]

    n_parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    # Optimizer
    if args.optimizer == 'SGD':
        optimizers = [torch.optim.SGD(params=models[i].parameters(), lr=args.learning_rate, momentum=0.9) for i in range(args.num_patients)]
    elif args.optimizer == 'Adam':
        optimizers = [torch.optim.Adam(params=models[i].parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay) for i in range(args.num_patients)]
    else:
        raise ValueError("Optimizer chosen not implemented!")

    schedulers = [MultiStepLR(optimizers[i], milestones=[args.epochs//2, args.epochs//4*3], gamma=0.1) for i in range(args.num_patients)]

    train_datasets, train_dist_datasets, valid_datasets = [], [], []

    for patient in ["P"+str(i) for i in range(1,args.num_patients+1)]:
        train_dataset = PatientDataset(features_path=args.features_path,
                                       dataset_path=args.dataset_path,
                                       mode='train', window_size=args.window_size, stride=args.stride, patient=patient)
        train_datasets.append(train_dataset)

        train_dist_datasets.append(PatientDataset(features_path=args.features_path,
                                       dataset_path=args.dataset_path,
                                       mode='train_dist', window_size=args.window_size, stride=args.stride, patient=patient))

        valid_datasets.append(PatientDataset(features_path=args.features_path,
                                       dataset_path=args.dataset_path,
                                       mode='val', scaler=train_dataset.scaler, window_size=args.window_size,
                                       stride=args.stride, patient=patient))

    # save scaler as pkl
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)



    #print('Length of train dataset:', len(train_dataset))
    #print('Length of valid dataset:', len(valid_dataset))

    # add a collate fn which ignores None
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    all_loaders = []
    for i in range(args.num_patients):
        loaders = {
            'train': torch.utils.data.DataLoader(train_datasets[i], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn),
            'val': torch.utils.data.DataLoader(valid_datasets[i], batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn),
            'train_distribution': torch.utils.data.DataLoader(train_dist_datasets[i], batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        }
        all_loaders.append(loaders)

    # Trainer
    trainer = Trainer(
        models=models,
        optims=optimizers,
        scheds=schedulers,
        loaders=all_loaders,
        args=args
    )

    trainer.train()





if __name__ == '__main__':
    main()
