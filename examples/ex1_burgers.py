from libs_path import *
from libs import *
torch.cuda.set_device(2)
import time
def main():
    
    ####
    # with northo 
#     thred = 1e-3
#     alpha = 0.3
#     beta = 1e-3
#     epsilon = 1e-4
#     lr = 1e-4

#      initial
#     v : 0.1* torch.randn
#     s,d:  rand    


     ####
    
    
    
    
    args = get_args_1d()
    k = 100
#     thred = 1e-2
#     alpha = 0.3
    beta = 1e-3
#     epsilon = 1e-4
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    kwargs = {'pin_memory': True} if cuda else {}
    get_seed(args.seed, printout=False)

    data_path = os.path.join(DATA_PATH, 'burgers_data_R10.mat')
    train_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=True,
                                   train_portion=0.5,
                                   data_path=data_path,)

    valid_dataset = BurgersDataset(subsample=args.subsample,
                                   train_data=False,
                                   valid_portion=100,
                                   data_path=data_path,)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, shuffle=False,
                              drop_last=False, **kwargs)

    print("train :", len(train_loader))
    print("valid :", len(valid_loader))
    sample = next(iter(train_loader))
    seq_len = 2**13//args.subsample

    print('='*20, 'Data loader batch', '='*20)
    for key in sample.keys():
        print(key, "\t", sample[key].shape)
    print('='*(40 + len('Data loader batch')+2))

    if is_interactive():
        u0 = sample['node']
        pos = sample['pos']
        u = sample['target']
        _, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
        axes = axes.reshape(-1)
        indexes = np.random.choice(range(4), size=4, replace=False)
        for i, ix in enumerate(indexes):
            axes[i].plot(pos[ix], u0[ix], label='input')
            axes[i].plot(pos[ix], u[ix, :, 0], label='target')
            axes[i].plot(pos[ix, 1:-1], u[ix, 1:-1, 1],
                         label='target derivative')
            axes[i].legend()

    with open(os.path.join(SRC_ROOT, 'config.yml')) as f:
        config = yaml.full_load(f)
    test_name = os.path.basename(__file__).split('.')[0]
    config = config[test_name]
    config['attn_norm'] = not args.layer_norm
    for arg in vars(args):
        if arg in config.keys():
            config[arg] = getattr(args, arg)
            
    config["seq_len"] = seq_len
    config["k"] = k
    
    get_seed(args.seed)
    torch.cuda.empty_cache()
    model = SimpleTransformer(**config)
    #model =ResNet(input_width = 1, layer_width = 64) 
    model = model.to(device)
    print(f"\nModel: {model.__name__}\t Number of params: {get_num_params(model)}")

    model_name, result_name = get_model_name(model='Burgers',
                                         num_encoder_layers=config['num_encoder_layers'],
                                         n_hidden=config['n_hidden'],
                                         attention_type=config['attention_type'],
                                         layer_norm=config['layer_norm'],
                                         grid_size=int(2**13//args.subsample),
                                         )
    print(f"Saving model and result in {MODEL_PATH}/{model_name}\n")

    epochs = args.epochs
#     lr = args.lr
    lr = 3e-4
    h = (1/2**13)*args.subsample
    tqdm_mode = 'epoch' if not args.show_batch else 'batch'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e4,
                           pct_start=0.2,
                           final_div_factor=1e4,
                           steps_per_epoch=len(train_loader), epochs=epochs)


#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    loss_func = WeightedL2Loss(regularizer=True, h=h, gamma=args.gamma)

    metric_func = WeightedL2Loss(regularizer=False, h=h)
    
    start = time.time()

    result = run_train(model, loss_func, metric_func,
                       train_loader, valid_loader,
                       optimizer, scheduler,
                       train_batch=train_batch_burgers,
                       validate_epoch=validate_epoch_burgers,
                       epochs=epochs,
                       patience=None,
                       tqdm_mode=tqdm_mode,
                       model_name=model_name,
                       result_name=result_name,
                       device=device,
                       L1 = False,
                       beta = beta)
    end = time.time()
    mean_time = (end-start)/epochs
    
    path = os.path.join(SRC_ROOT, 'high_hidden_result')
    train_time_path = os.path.join(path, "burgers_%s_training_time_per_epoch.npy"%args.attention_type)
    np.save(train_time_path, mean_time)
    
   
    if args.attention_type == "Lrk":
        param = torch.load(os.path.join(MODEL_PATH, model_name))
        
        v_temp = param["v"]
        v = v_temp**2
        print("v", torch.sort(v, descending=True))
        param["v"] = param["v"][v>epsilon]
        param["s"] = param["s"][:, v>epsilon]
        param["d"] = param["d"][v>epsilon, :]
        
        k = len(param["v"])
        config["k"] = k
        config["attention_type"] = "Lrk"
        config["seq_len"] = int(8192/args.subsample)
        print("the value of k after updating:", k)
        torch.save(param,os.path.join(MODEL_PATH, model_name))
        
        del model
        model = SimpleTransformer(**config)
        model = model.cuda()
        model.load_state_dict(param)
#         torch.save(model.state_dict(),os.path.join(MODEL_PATH, model_name))
        model.eval()
        val_metric = validate_epoch_burgers(model, metric_func, valid_loader, device)
        print(f" model's validation metric after pruning in this run: {val_metric}")
    
    else:
    
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, model_name)))
        model.eval()
        val_metric = validate_epoch_burgers(model, metric_func, valid_loader, device)
        print(f"\nBest model's validation metric in this run: {val_metric}")
    
    
    
    training_curve_path = os.path.join(path, "burgers_%s_training_loss.npy"%args.attention_type)
    valid_curve_path = os.path.join(path, "burgers_%s_valid_loss.npy"%args.attention_type)
    
    
    train_loss = result["loss_train"]
    val_loss = result["loss_val"]
    np.save(training_curve_path, train_loss)
    np.save(valid_curve_path, val_loss)
    
    if args.attention_type == "fourier":
        sample = next(iter(valid_loader))
        a = sample['node']
        pos = sample['pos']
        u = sample['target']
        grid = sample['grid']
        with torch.no_grad():
            model.eval()
            out_ = model(a.to(device), None, pos.to(device), grid.to(device))
            out = out_['attn_weights']


        eigenvalue_path = os.path.join(path, 'burgers attention')
        torch.save(out, eigenvalue_path) 

    

if __name__ == '__main__':
    main()
